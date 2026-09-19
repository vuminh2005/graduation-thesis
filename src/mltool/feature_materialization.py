"""Phase-3 prepared-input validation and atomic FeatureSet materialization."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
from uuid import uuid4

import pandas as pd

from mltool.config import FeaturePluginConfig, FeatureSetConfig, MLToolConfig
from mltool.feature_plugins import (
    FeaturePluginError,
    fit_and_generate_features,
)


class FeatureMaterializationError(ValueError):
    """An expected prepared-data, plugin, FeatureSet, or artifact failure."""


@dataclass
class PreparedFeatureInput:
    path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    source_fingerprint: str


@dataclass
class MaterializedFeatureSet:
    name: str
    base_feature_count: int
    plugin_names: list[str]
    generated_feature_count: int
    final_feature_count: int
    rows: dict[str, int]
    frames: dict[str, pd.DataFrame]
    manifest: dict[str, Any]
    fitted_plugins: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureMaterializationResult:
    project_root: Path
    prepared_path: Path
    output_path: Path
    prepared_rows: dict[str, int]
    feature_sets: list[MaterializedFeatureSet]

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)

    def render(self) -> str:
        lines = [
            "MLTool feature materialization",
            "",
            "Prepared input",
            f"  {self._display_path(self.prepared_path)}",
            f"  Train: {self.prepared_rows['train']} rows",
            f"  Validation: {self.prepared_rows['validation']} rows",
            f"  Test: {self.prepared_rows['test']} rows",
            "",
            "Feature sets",
        ]
        total = len(self.feature_sets)
        for index, feature_set in enumerate(self.feature_sets, start=1):
            plugin_label = ", ".join(feature_set.plugin_names) or "0"
            lines.extend(
                [
                    "",
                    f"  [{index}/{total}] {feature_set.name}",
                    f"        Base features: {feature_set.base_feature_count}",
                    f"        Plugins: {plugin_label}",
                    f"        Generated features: {feature_set.generated_feature_count}",
                    f"        Final features: {feature_set.final_feature_count}",
                    "        MATERIALIZED",
                ]
            )
        lines.extend(
            [
                "",
                "Outputs",
                f"  {self._display_path(self.output_path)}/",
                "",
                "Result: MATERIALIZED",
            ]
        )
        return "\n".join(lines)


def render_feature_error(message: str) -> str:
    return "\n".join(
        [
            "MLTool feature materialization",
            "",
            "Errors",
            f"  x {message}",
            "",
            "Result: FAILED",
        ]
    )


def _fingerprint(path: Path) -> str:
    if not path.is_file():
        raise FeatureMaterializationError(
            f"configured source dataset was not found: {path}; run \"mltool prepare\" again"
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FeatureMaterializationError(
            f"could not fingerprint configured source dataset {path}: {exc}"
        ) from exc
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FeatureMaterializationError(
            f'prepared artifacts were not found; run "mltool prepare" first ({path})'
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureMaterializationError(
            f'prepared manifest is unreadable; run "mltool prepare" again: {exc}'
        ) from exc
    if not isinstance(value, dict):
        raise FeatureMaterializationError(
            'prepared manifest is invalid; run "mltool prepare" again'
        )
    return value


def load_prepared_feature_input(config: MLToolConfig) -> PreparedFeatureInput:
    prepared_path = config.config_path.parent / ".mltool/prepared"
    manifest_path = prepared_path / "manifest.json"
    manifest = _read_manifest(manifest_path)

    manifest_task = manifest.get("task")
    if not isinstance(manifest_task, dict) or (
        manifest_task.get("type") != config.task.type
        or manifest_task.get("target") != config.task.target
    ):
        raise FeatureMaterializationError(
            'prepared task/target does not match the current config; run "mltool prepare" again'
        )
    manifest_source = manifest.get("source")
    prepared_fingerprint = (
        manifest_source.get("fingerprint") if isinstance(manifest_source, dict) else None
    )
    if not isinstance(prepared_fingerprint, str) or not prepared_fingerprint:
        raise FeatureMaterializationError(
            'prepared manifest has no source fingerprint; run "mltool prepare" again'
        )
    current_fingerprint = _fingerprint(config.data.path)
    if current_fingerprint != prepared_fingerprint:
        raise FeatureMaterializationError(
            'configured source dataset changed after preparation; run "mltool prepare" again'
        )

    frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test"):
        path = prepared_path / f"{split_name}.parquet"
        if not path.is_file():
            raise FeatureMaterializationError(
                f'prepared split is missing: {path}; run "mltool prepare" again'
            )
        try:
            frame = pd.read_parquet(path, engine="pyarrow")
        except Exception as exc:
            raise FeatureMaterializationError(
                f'could not read prepared {split_name} split {path}: {exc}; '
                'run "mltool prepare" again'
            ) from exc
        if frame.columns.duplicated().any():
            raise FeatureMaterializationError(
                f'prepared {split_name} split contains duplicate columns; '
                'run "mltool prepare" again'
            )
        if config.task.target not in frame.columns:
            raise FeatureMaterializationError(
                f'configured target "{config.task.target}" is missing from prepared '
                f'{split_name} split; run "mltool prepare" again'
            )
        frames[split_name] = frame

    expected_columns = frames["train"].columns.tolist()
    expected_dtypes = [str(dtype) for dtype in frames["train"].dtypes]
    for split_name in ("validation", "test"):
        actual_columns = frames[split_name].columns.tolist()
        actual_dtypes = [str(dtype) for dtype in frames[split_name].dtypes]
        if actual_columns != expected_columns or actual_dtypes != expected_dtypes:
            raise FeatureMaterializationError(
                f"prepared train/{split_name} schemas are incompatible; "
                'run "mltool prepare" again'
            )

    return PreparedFeatureInput(
        path=prepared_path,
        manifest_path=manifest_path,
        manifest=manifest,
        train=frames["train"],
        validation=frames["validation"],
        test=frames["test"],
        source_fingerprint=prepared_fingerprint,
    )


def _selected_source_columns(
    spec: FeatureSetConfig,
    available: list[str],
    target: str,
) -> list[str]:
    if spec.source_columns == ["*"]:
        return [column for column in available if column != target]
    if target in spec.source_columns:
        raise FeatureMaterializationError(
            f'feature set "{spec.name}" may not select configured target column "{target}"'
        )
    missing = [column for column in spec.source_columns if column not in available]
    if missing:
        raise FeatureMaterializationError(
            f'feature set "{spec.name}" requests missing source column "{missing[0]}"'
        )
    return list(spec.source_columns)


def _plugin_manifest(
    spec: FeaturePluginConfig,
    generated_columns: list[str],
    resolved_entrypoint: str,
) -> dict[str, Any]:
    return {
        "name": spec.name,
        "entrypoint": spec.entrypoint,
        "resolved_entrypoint": resolved_entrypoint,
        "params": spec.params,
        "generated_columns": generated_columns,
    }


def _build_feature_set(
    config: MLToolConfig,
    prepared: PreparedFeatureInput,
    spec: FeatureSetConfig,
    plugin_catalog: dict[str, FeaturePluginConfig],
) -> MaterializedFeatureSet:
    return build_feature_set_from_frames(
        config,
        {"train": prepared.train, "validation": prepared.validation, "test": prepared.test},
        "train",
        spec,
        plugin_catalog,
    )


def build_feature_set_from_frames(
    config: MLToolConfig,
    prepared_frames: dict[str, pd.DataFrame],
    fit_split: str,
    spec: FeatureSetConfig,
    plugin_catalog: dict[str, FeaturePluginConfig],
) -> MaterializedFeatureSet:
    """Build one FeatureSet, fitting every plugin once on ``fit_split`` only.

    Phase 3 fits on train; Phase 6 refits on train+validation. The plugin
    instantiate/fit/transform contract is identical in both.
    """
    target = config.task.target
    split_names = list(prepared_frames)
    source_columns = _selected_source_columns(
        spec, prepared_frames[fit_split].columns.tolist(), target
    )
    base_frames = {
        split_name: frame.loc[:, source_columns].copy(deep=True)
        for split_name, frame in prepared_frames.items()
    }
    generated_by_split: dict[str, list[pd.DataFrame]] = {name: [] for name in split_names}
    occupied_columns = set(source_columns)
    plugin_manifests: list[dict[str, Any]] = []
    fitted_plugins: dict[str, Any] = {}
    lineage: dict[str, dict[str, str]] = {
        column: {"source": "base"} for column in source_columns
    }

    for plugin_name in spec.plugins:
        plugin_spec = plugin_catalog[plugin_name]
        generated_frames, generated_columns, resolved_entrypoint, fitted = fit_and_generate_features(
            plugin_spec,
            config_path=config.config_path,
            target=target,
            fit_split=fit_split,
            base_frames=base_frames,
        )
        collisions = [
            column for column in generated_columns if column in occupied_columns
        ]
        if collisions:
            raise FeatureMaterializationError(
                f'feature plugin "{plugin_name}" generated colliding column '
                f'"{collisions[0]}" in feature set "{spec.name}"'
            )
        occupied_columns.update(generated_columns)
        fitted_plugins[plugin_name] = fitted
        for column in generated_columns:
            lineage[column] = {"source": "plugin", "plugin": plugin_name}
        for split_name in split_names:
            generated_by_split[split_name].append(generated_frames[split_name])
        plugin_manifests.append(
            _plugin_manifest(plugin_spec, generated_columns, resolved_entrypoint)
        )

    final_feature_columns = list(source_columns)
    for plugin_manifest in plugin_manifests:
        final_feature_columns.extend(plugin_manifest["generated_columns"])

    final_frames: dict[str, pd.DataFrame] = {}
    for split_name in split_names:
        pieces = [base_frames[split_name], *generated_by_split[split_name]]
        features = pd.concat(pieces, axis=1)
        if features.columns.tolist() != final_feature_columns:
            raise FeatureMaterializationError(
                f'feature set "{spec.name}" produced inconsistent final schema '
                f'for split "{split_name}"'
            )
        result = features.copy(deep=True)
        result[target] = prepared_frames[split_name][target].copy(deep=True)
        final_frames[split_name] = result

    expected_schema = final_frames[fit_split].columns.tolist()
    for split_name in split_names:
        if final_frames[split_name].columns.tolist() != expected_schema:
            raise FeatureMaterializationError(
                f'feature set "{spec.name}" final {fit_split}/{split_name} schemas differ'
            )

    rows = {split_name: len(frame) for split_name, frame in final_frames.items()}
    manifest = {
        "name": spec.name,
        "source_columns": source_columns,
        "plugins": plugin_manifests,
        "final_feature_columns": final_feature_columns,
        "target": target,
        "rows": rows,
        "lineage": lineage,
    }
    return MaterializedFeatureSet(
        name=spec.name,
        base_feature_count=len(source_columns),
        plugin_names=list(spec.plugins),
        generated_feature_count=len(final_feature_columns) - len(source_columns),
        final_feature_count=len(final_feature_columns),
        rows=rows,
        frames=final_frames,
        manifest=manifest,
        fitted_plugins=fitted_plugins,
    )


def _commit_staged_directory(staging: Path, output_path: Path) -> None:
    backup: Path | None = None
    if output_path.exists():
        backup = output_path.parent / f".features-backup-{uuid4().hex}"
        output_path.rename(backup)
    try:
        staging.rename(output_path)
    except Exception:
        if backup is not None and backup.exists() and not output_path.exists():
            backup.rename(output_path)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def materialize_feature_sets(config: MLToolConfig) -> FeatureMaterializationResult:
    prepared = load_prepared_feature_input(config)
    project_root = config.config_path.parent
    workspace = project_root / ".mltool"
    output_path = workspace / "features"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".features-staging-", dir=workspace))
    except OSError as exc:
        raise FeatureMaterializationError(
            f"could not create feature workspace: {exc}"
        ) from exc

    catalog = {plugin.name: plugin for plugin in config.features.plugins}
    feature_sets: list[MaterializedFeatureSet] = []
    try:
        for spec in config.features.sets:
            materialized = _build_feature_set(config, prepared, spec, catalog)
            set_staging = staging / spec.name
            set_staging.mkdir()
            for split_name in ("train", "validation", "test"):
                materialized.frames[split_name].to_parquet(
                    set_staging / f"{split_name}.parquet", index=False
                )
            (set_staging / "manifest.json").write_text(
                json.dumps(materialized.manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            feature_sets.append(materialized)

        global_manifest = {
            "prepared_input": {
                "path": str(prepared.path),
                "manifest": str(prepared.manifest_path),
            },
            "source_dataset_fingerprint": prepared.source_fingerprint,
            "task": {
                "type": config.task.type,
                "target": config.task.target,
            },
            "target": config.task.target,
            "prepared_feature_columns": [
                column for column in prepared.train.columns if column != config.task.target
            ],
            "rows": {
                "train": len(prepared.train),
                "validation": len(prepared.validation),
                "test": len(prepared.test),
            },
            "feature_sets": [
                {
                    "name": feature_set.name,
                    "path": str(output_path / feature_set.name),
                    "feature_count": feature_set.final_feature_count,
                    "rows": feature_set.rows,
                }
                for feature_set in feature_sets
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(global_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _commit_staged_directory(staging, output_path)
    except FeaturePluginError as exc:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise FeatureMaterializationError(str(exc)) from exc
    except FeatureMaterializationError:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise FeatureMaterializationError(
            f"could not materialize feature artifacts: {exc}"
        ) from exc

    return FeatureMaterializationResult(
        project_root=project_root,
        prepared_path=prepared.path,
        output_path=output_path,
        prepared_rows={
            "train": len(prepared.train),
            "validation": len(prepared.validation),
            "test": len(prepared.test),
        },
        feature_sets=feature_sets,
    )
