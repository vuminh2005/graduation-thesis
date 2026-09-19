"""Phase-7 MLTool-owned local model registry: versioned copies of ``.mltool/final/``."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from mltool import __version__
from mltool.config import MLToolConfig
from mltool.finalize import FinalizeError, load_persisted_final
from mltool.state import utc_now
from mltool.training import _commit_staged_directory


class RegistryError(ValueError):
    """An expected registry precondition or artifact problem."""


@dataclass
class RegisteredModel:
    version: int
    path: Path
    metadata: dict[str, Any]

    def render(self) -> str:
        meta = self.metadata
        lines = [
            "MLTool model registry",
            "",
            f"Registered version {self.version}",
            f"  {self.path}",
            "",
            f"Selected: {meta['selected']['candidate_id']}",
            f"Config fingerprint: {meta['config_fingerprint'][:16]}",
            f"Git commit: {meta['git_commit'] or 'n/a'}",
        ]
        if meta.get("warning"):
            lines.extend(["", "Warnings", f"  ! {meta['warning']}"])
        lines.extend(["", "Result: REGISTERED"])
        return "\n".join(lines)


def registry_path(project_root: Path) -> Path:
    return project_root / ".mltool" / "registry"


def list_versions(project_root: Path) -> list[int]:
    root = registry_path(project_root)
    if not root.is_dir():
        return []
    return sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit())


def load_metadata(project_root: Path, version: int) -> dict[str, Any]:
    path = registry_path(project_root) / str(version) / "metadata.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"registry metadata for version {version} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistryError(f"registry metadata for version {version} is invalid")
    return value


def _git_commit(project_root: Path) -> str | None:
    """Best-effort: None outside a git repository or when git is unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None


def config_fingerprint(config: MLToolConfig) -> str:
    return hashlib.sha256(config.config_path.read_bytes()).hexdigest()


def register_final(config: MLToolConfig) -> RegisteredModel:
    project_root = config.config_path.parent
    final = project_root / ".mltool" / "final"
    if not (final / "manifest.json").is_file() or not (final / "result.json").is_file():
        raise RegistryError('final artifacts were not found; run "mltool finalize" first')
    try:
        persisted = load_persisted_final(config)
    except FinalizeError as exc:
        raise RegistryError(str(exc)) from exc
    result, manifest = persisted.result, persisted.manifest

    root = registry_path(project_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".registry-staging-", dir=root))
    except OSError as exc:
        raise RegistryError(f"could not create registry workspace: {exc}") from exc

    try:
        for name in ("predictor", "feature_plugins"):
            if (final / name).is_dir():
                shutil.copytree(final / name, staging / name)
        for name in ("preprocessor.pkl", "result.json", "manifest.json"):
            if (final / name).is_file():
                shutil.copy2(final / name, staging / name)
        version = (max(list_versions(project_root), default=0)) + 1
        metadata = {
            "version": version,
            "registered_at": utc_now(),
            "mltool_version": __version__,
            "git_commit": _git_commit(project_root),
            "config_fingerprint": config_fingerprint(config),
            "source_dataset_fingerprint": manifest.get("source_dataset_fingerprint"),
            "selected": manifest["selected"],
            "best_hyperparameters": result["best_hyperparameters"],
            "seed": result.get("seed"),
            "effective_seed": result.get("effective_seed"),
            "primary_metric": result["primary_metric"],
            "test_metrics": result["metrics"],
            "selected_validation_score": result["selected_validation_score"],
            "autogluon_version": result.get("autogluon_version"),
            "artifacts": sorted([p.name for p in staging.iterdir()] + ["metadata.json"]),
            "warning": persisted.warning,
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        # Versions are monotonic; the rename refuses to overwrite an existing version.
        target = root / str(version)
        if target.exists():
            raise RegistryError(f"registry version {version} already exists")
        _commit_staged_directory(staging, target)
    except RegistryError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise RegistryError(f"could not write registry entry: {exc}") from exc
    return RegisteredModel(version=version, path=target, metadata=metadata)
