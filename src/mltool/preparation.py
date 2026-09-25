"""Phase-2 preparation orchestration and local artifact materialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import uuid4

import cloudpickle

from mltool.config import MLToolConfig
from mltool.data import LoadedDataset
from mltool.preprocessing import (
    PreprocessedSplits,
    PreprocessingError,
    preprocess_splits,
    preprocessor_source_sha256,
)
from mltool.splitting import DatasetSplits, SplitError, split_dataset


class PreparationError(ValueError):
    """An expected Phase-2 split, preprocessing, or artifact failure."""


@dataclass
class PreparationResult:
    project_root: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    manifest_path: Path
    preprocessor_path: Path | None
    train_rows: int
    validation_rows: int
    test_rows: int
    stratified: bool
    random_seed: int
    warning_messages: list[str]
    preprocessor_name: str | None = None

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)

    def render(self) -> str:
        warning_count = len(self.warning_messages)
        warning_label = "warning" if warning_count == 1 else "warnings"
        lines = [
            "MLTool preparation",
            "",
            "Validation",
            f"  VALID with {warning_count} {warning_label}",
        ]
        lines.extend(f"  ! {warning}" for warning in self.warning_messages)
        lines.extend(
            [
                "",
                "Split",
                f'  Strategy: {"stratified" if self.stratified else "not stratified"}',
                f"  Random seed: {self.random_seed}",
                f"  Train: {self.train_rows} rows",
                f"  Validation: {self.validation_rows} rows",
                f"  Test: {self.test_rows} rows",
                "",
                "External preprocessing",
            ]
        )
        if self.preprocessor_name is None:
            lines.append("  Disabled")
        else:
            lines.extend(
                [
                    f"  {self.preprocessor_name}",
                    "  fit: train features only",
                    "  transformed: train, validation, test",
                    f"  artifact: {self._display_path(self.preprocessor_path)}",
                ]
            )
        lines.extend(
            [
                "",
                "Outputs",
                f"  {self._display_path(self.train_path)}",
                f"  {self._display_path(self.validation_path)}",
                f"  {self._display_path(self.test_path)}",
                f"  {self._display_path(self.manifest_path)}",
                "",
                "Result: PREPARED",
            ]
        )
        return "\n".join(lines)


def render_preparation_error(message: str) -> str:
    return "\n".join(
        ["MLTool preparation", "", "Errors", f"  x {message}", "", "Result: FAILED"]
    )


def _manifest(
    config: MLToolConfig,
    dataset: LoadedDataset,
    splits: DatasetSplits,
    prepared: PreprocessedSplits,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "source": {
            "path": str(dataset.path),
            "format": dataset.format,
            "fingerprint": dataset.fingerprint,
        },
        "task": {"type": config.task.type, "target": config.task.target},
        "split": {
            "random_seed": config.split.random_seed,
            "stratified": splits.stratified,
            "requested_validation_ratio": config.split.validation_ratio,
            "requested_test_ratio": config.split.test_ratio,
            "train_rows": len(prepared.train),
            "validation_rows": len(prepared.validation),
            "test_rows": len(prepared.test),
        },
        "preprocessing": {"enabled": config.preprocessing.external.enabled},
        # relative to this manifest's directory
        "outputs": {
            "train": "train.parquet",
            "validation": "validation.parquet",
            "test": "test.parquet",
        },
    }
    if config.preprocessing.external.enabled:
        manifest["preprocessing"].update(
            {
                "entrypoint": config.preprocessing.external.entrypoint,
                "resolved_entrypoint": prepared.resolved_entrypoint,
                "params": config.preprocessing.external.params,
                # editing the preprocessor file makes these artifacts stale
                "source_sha256": preprocessor_source_sha256(
                    config.preprocessing.external, config.config_path
                ),
                "artifact": "preprocessor.pkl",
            }
        )
    return manifest


def _commit_staged_directory(staging: Path, output_dir: Path) -> None:
    backup: Path | None = None
    if output_dir.exists():
        backup = output_dir.parent / f".prepared-backup-{uuid4().hex}"
        output_dir.rename(backup)
    try:
        staging.rename(output_dir)
    except Exception:
        if backup is not None and backup.exists() and not output_dir.exists():
            backup.rename(output_dir)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _materialize(
    config: MLToolConfig,
    dataset: LoadedDataset,
    splits: DatasetSplits,
    prepared: PreprocessedSplits,
) -> tuple[Path, Path, Path, Path, Path | None]:
    project_root = config.config_path.parent
    workspace = project_root / ".mltool"
    output_dir = workspace / "prepared"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".prepared-staging-", dir=workspace))
    except OSError as exc:
        raise PreparationError(f"could not create preparation workspace: {exc}") from exc

    preprocessor_path: Path | None = None
    try:
        prepared.train.to_parquet(staging / "train.parquet", index=False)
        prepared.validation.to_parquet(staging / "validation.parquet", index=False)
        prepared.test.to_parquet(staging / "test.parquet", index=False)

        if prepared.fitted_preprocessor is not None:
            preprocessor_path = output_dir / "preprocessor.pkl"
            try:
                with (staging / "preprocessor.pkl").open("wb") as stream:
                    cloudpickle.dump(prepared.fitted_preprocessor, stream)
            except (Exception, SystemExit) as exc:
                raise PreparationError(
                    f"could not serialize fitted external preprocessor: {exc}"
                ) from exc

        manifest = _manifest(config, dataset, splits, prepared)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _commit_staged_directory(staging, output_dir)
    except PreparationError:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise PreparationError(f"could not write prepared artifacts: {exc}") from exc

    return (
        output_dir / "train.parquet",
        output_dir / "validation.parquet",
        output_dir / "test.parquet",
        output_dir / "manifest.json",
        preprocessor_path,
    )


def prepare_dataset(
    config: MLToolConfig,
    dataset: LoadedDataset,
    *,
    warning_messages: list[str],
) -> PreparationResult:
    try:
        splits = split_dataset(
            dataset.frame,
            target=config.task.target,
            task_type=config.task.type,
            config=config.split,
        )
        prepared = preprocess_splits(
            splits,
            target=config.task.target,
            config=config.preprocessing.external,
            config_path=config.config_path,
        )
    except (SplitError, PreprocessingError) as exc:
        raise PreparationError(str(exc)) from exc

    train_path, validation_path, test_path, manifest_path, preprocessor_path = (
        _materialize(config, dataset, splits, prepared)
    )
    return PreparationResult(
        project_root=config.config_path.parent,
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        manifest_path=manifest_path,
        preprocessor_path=preprocessor_path,
        train_rows=len(prepared.train),
        validation_rows=len(prepared.validation),
        test_rows=len(prepared.test),
        stratified=splits.stratified,
        random_seed=config.split.random_seed,
        warning_messages=list(warning_messages),
        preprocessor_name=prepared.preprocessor_name,
    )
