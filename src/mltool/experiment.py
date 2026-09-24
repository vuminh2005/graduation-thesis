"""Phase-4 feature-artifact validation and deterministic experiment planning."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mltool.config import FeatureSetConfig, MLToolConfig, ModelConfig
from mltool.feature_materialization import (
    FeatureMaterializationError,
    prepared_artifacts_fingerprint,
)


class ExperimentError(ValueError):
    """An expected configuration or feature-artifact problem."""


@dataclass(frozen=True)
class FeatureSetArtifact:
    name: str
    path: Path
    train_path: Path
    validation_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    train_sha256: str
    validation_sha256: str
    feature_columns: list[str]
    train_rows: int
    validation_rows: int


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    feature_set: FeatureSetArtifact
    model: ModelConfig


@dataclass(frozen=True)
class ExperimentPlan:
    config: MLToolConfig
    feature_manifest_path: Path
    feature_manifest: dict[str, Any]
    feature_sets: list[FeatureSetArtifact]
    candidates: list[CandidateSpec]

    def render(self) -> str:
        task_label = {
            "binary": "binary classification",
            "multiclass": "multiclass classification",
            "regression": "regression",
        }[self.config.task.type]
        lines = [
            "MLTool training plan",
            "",
            "Task",
            f"  {task_label}",
            "",
            "Primary metric",
            f"  {self.config.evaluation.primary_metric} ({self.config.evaluation.direction})",
            "",
            "Feature sets",
        ]
        lines.extend(
            f"  {index}. {artifact.name}"
            for index, artifact in enumerate(self.feature_sets, start=1)
        )
        lines.extend(["", "Models"])
        lines.extend(
            f"  {index}. {model.name} [{model.family}]"
            for index, model in enumerate(self.config.models, start=1)
        )
        lines.extend(["", "Candidates"])
        lines.extend(
            f"  {index}. {candidate.feature_set.name} x {candidate.model.name}"
            for index, candidate in enumerate(self.candidates, start=1)
        )
        lines.extend(
            [
                "",
                f"Total candidates: {len(self.candidates)}",
                "",
                "HPO: disabled",
                "Test data: not used",
            ]
        )
        return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ExperimentError(f"could not fingerprint artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ExperimentError(
            f'feature artifacts are missing ({path}); run "mltool features" first'
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(
            f'{label} is unreadable; run "mltool features" again: {exc}'
        ) from exc
    if not isinstance(value, dict):
        raise ExperimentError(f'{label} is invalid; run "mltool features" again')
    return value


def _stale(message: str) -> ExperimentError:
    return ExperimentError(
        f'feature artifacts are stale ({message}); run "mltool features" again'
    )


def _current_source_fingerprint(config: MLToolConfig) -> str:
    if not config.data.path.is_file():
        raise _stale(f"configured source dataset is missing: {config.data.path}")
    return sha256_file(config.data.path)


def _validate_prepared_provenance(
    config: MLToolConfig, global_manifest: dict[str, Any]
) -> None:
    """The prepared artifacts these features were built from must still be the current ones.

    Re-running ``prepare`` alone (a changed split seed or ratio) leaves the raw
    dataset fingerprint intact, so without this every downstream phase would
    keep trusting features derived from a superseded split.
    """
    recorded = global_manifest.get("prepared_artifacts_fingerprint")
    if not isinstance(recorded, str) or not recorded:
        # Written by older MLTool versions: provenance cannot be proven.
        raise _stale(
            "global feature manifest predates prepared-artifact fingerprinting"
        )
    try:
        current = prepared_artifacts_fingerprint(config.config_path.parent / ".mltool/prepared")
    except FeatureMaterializationError as exc:
        raise ExperimentError(str(exc)) from exc
    if recorded != current:
        raise _stale("prepared artifacts changed after materialization")


def _feature_set_entries(global_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = global_manifest.get("feature_sets")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise _stale("global feature manifest has no valid feature-set list")
    return entries


def _expected_source_columns(
    spec: FeatureSetConfig,
    prepared_feature_columns: list[str],
) -> list[str]:
    if spec.source_columns == ["*"]:
        return list(prepared_feature_columns)
    return list(spec.source_columns)


def feature_set_recipe(
    config: MLToolConfig,
    spec: FeatureSetConfig,
    prepared_feature_columns: list[str],
) -> dict[str, Any]:
    """Everything about a FeatureSet that decides which columns it materializes.

    The single definition of a "recipe": ``_validate_recipe`` checks a
    materialized manifest against it, and ``finalize`` fingerprints it so a
    later recipe change can be detected without re-reading ``features/``.
    """
    catalog = {plugin.name: plugin for plugin in config.features.plugins}
    return {
        "name": spec.name,
        "target": config.task.target,
        "source_columns": _expected_source_columns(spec, prepared_feature_columns),
        "plugin_inputs": list(spec.plugin_inputs),
        "plugins": [
            {
                "name": name,
                "entrypoint": catalog[name].entrypoint,
                "params": catalog[name].params,
            }
            for name in spec.plugins
        ],
    }


def recipe_fingerprint(recipe: dict[str, Any]) -> str:
    payload = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_feature_set_recipe(config: MLToolConfig, name: str) -> dict[str, Any]:
    """The configured recipe for one FeatureSet, with ``*`` source columns resolved."""
    for spec in config.features.sets:
        if spec.name == name:
            break
    else:
        raise ExperimentError(f'feature set "{name}" is no longer configured')
    global_manifest = _read_json(
        config.config_path.parent / ".mltool/features/manifest.json",
        "global feature manifest",
    )
    prepared_feature_columns = global_manifest.get("prepared_feature_columns")
    if not isinstance(prepared_feature_columns, list) or not all(
        isinstance(column, str) for column in prepared_feature_columns
    ):
        raise _stale("global feature manifest has no resolved prepared feature schema")
    return feature_set_recipe(config, spec, list(prepared_feature_columns))


def _validate_recipe(
    config: MLToolConfig,
    spec: FeatureSetConfig,
    manifest: dict[str, Any],
    prepared_feature_columns: list[str],
) -> list[str]:
    recipe = feature_set_recipe(config, spec, prepared_feature_columns)
    if manifest.get("name") != recipe["name"] or manifest.get("target") != recipe["target"]:
        raise _stale(f'feature set "{spec.name}" name/target does not match config')
    if manifest.get("source_columns") != recipe["source_columns"]:
        raise _stale(f'feature set "{spec.name}" source columns changed')
    # Missing key: materialized before plugin_inputs existed, which is only
    # equivalent to a set that does not configure any.
    if manifest.get("plugin_inputs", []) != recipe["plugin_inputs"]:
        raise _stale(f'feature set "{spec.name}" plugin inputs changed')

    plugins = manifest.get("plugins")
    if not isinstance(plugins, list):
        raise _stale(f'feature set "{spec.name}" plugin metadata is invalid')
    if [plugin.get("name") for plugin in plugins if isinstance(plugin, dict)] != spec.plugins:
        raise _stale(f'feature set "{spec.name}" plugin sequence changed')
    for plugin_entry, expected in zip(plugins, recipe["plugins"], strict=True):
        if not isinstance(plugin_entry, dict):
            raise _stale(f'feature set "{spec.name}" plugin metadata is invalid')
        if (
            plugin_entry.get("entrypoint") != expected["entrypoint"]
            or plugin_entry.get("params") != expected["params"]
        ):
            raise _stale(
                f'feature set "{spec.name}" plugin "{expected["name"]}" configuration changed'
            )

    final_columns = manifest.get("final_feature_columns")
    if not isinstance(final_columns, list) or not final_columns or not all(
        isinstance(column, str) for column in final_columns
    ):
        raise _stale(f'feature set "{spec.name}" final schema metadata is invalid')
    return list(final_columns)


def _read_training_split(
    path: Path,
    *,
    feature_set: str,
    split: str,
    expected_columns: list[str],
    target: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise ExperimentError(
            f'feature artifact is missing: {path}; run "mltool features" again'
        )
    try:
        frame = pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:
        raise ExperimentError(
            f'could not read feature set "{feature_set}" {split} artifact: {exc}; '
            'run "mltool features" again'
        ) from exc
    if frame.columns.duplicated().any():
        raise _stale(f'feature set "{feature_set}" {split} has duplicate columns')
    if frame.columns.tolist() != [*expected_columns, target]:
        raise _stale(f'feature set "{feature_set}" {split} schema differs from its manifest')
    return frame


def load_feature_artifacts(config: MLToolConfig) -> tuple[Path, dict[str, Any], list[FeatureSetArtifact]]:
    root = config.config_path.parent
    features_path = root / ".mltool/features"
    global_manifest_path = features_path / "manifest.json"
    global_manifest = _read_json(global_manifest_path, "global feature manifest")

    if global_manifest.get("task") != {
        "type": config.task.type,
        "target": config.task.target,
    }:
        raise _stale("configured task/target changed")
    prepared_feature_columns = global_manifest.get("prepared_feature_columns")
    if not isinstance(prepared_feature_columns, list) or not all(
        isinstance(column, str) for column in prepared_feature_columns
    ):
        raise _stale("global feature manifest has no resolved prepared feature schema")
    source_fingerprint = global_manifest.get("source_dataset_fingerprint")
    if source_fingerprint != _current_source_fingerprint(config):
        raise _stale("configured source dataset fingerprint changed")
    _validate_prepared_provenance(config, global_manifest)

    entries = _feature_set_entries(global_manifest)
    configured_names = [spec.name for spec in config.features.sets]
    materialized_names = [entry.get("name") for entry in entries]
    if materialized_names != configured_names:
        raise _stale("configured FeatureSet list/order changed")

    artifacts: list[FeatureSetArtifact] = []
    for spec in config.features.sets:
        set_path = features_path / spec.name
        manifest_path = set_path / "manifest.json"
        manifest = _read_json(manifest_path, f'feature set "{spec.name}" manifest')
        final_columns = _validate_recipe(
            config, spec, manifest, list(prepared_feature_columns)
        )
        train_path = set_path / "train.parquet"
        validation_path = set_path / "validation.parquet"
        train = _read_training_split(
            train_path,
            feature_set=spec.name,
            split="train",
            expected_columns=final_columns,
            target=config.task.target,
        )
        validation = _read_training_split(
            validation_path,
            feature_set=spec.name,
            split="validation",
            expected_columns=final_columns,
            target=config.task.target,
        )
        if [str(dtype) for dtype in train.dtypes] != [str(dtype) for dtype in validation.dtypes]:
            raise _stale(f'feature set "{spec.name}" train/validation dtypes differ')
        rows = manifest.get("rows")
        if not isinstance(rows, dict) or (
            rows.get("train") != len(train) or rows.get("validation") != len(validation)
        ):
            raise _stale(f'feature set "{spec.name}" row counts differ from its manifest')
        artifacts.append(
            FeatureSetArtifact(
                name=spec.name,
                path=set_path,
                train_path=train_path,
                validation_path=validation_path,
                manifest_path=manifest_path,
                manifest=manifest,
                train_sha256=sha256_file(train_path),
                validation_sha256=sha256_file(validation_path),
                feature_columns=final_columns,
                train_rows=len(train),
                validation_rows=len(validation),
            )
        )
    return global_manifest_path, global_manifest, artifacts


def build_experiment_plan(config: MLToolConfig) -> ExperimentPlan:
    if not config.models:
        raise ExperimentError(
            'no models are configured; add a non-empty "models" list to mltool.yaml'
        )
    for model in config.models:
        if model.family == "SKLEARN":
            # Build one instance now, so a contract violation (e.g. an SVC
            # without probability=True) is a config error before any fit.
            from mltool.custom_models import validate

            validate(model, config.task.type)
    manifest_path, manifest, feature_sets = load_feature_artifacts(config)
    candidates = [
        CandidateSpec(
            candidate_id=f"{feature_set.name}__{model.name}",
            feature_set=feature_set,
            model=model,
        )
        for feature_set in feature_sets
        for model in config.models
    ]
    return ExperimentPlan(
        config=config,
        feature_manifest_path=manifest_path,
        feature_manifest=manifest,
        feature_sets=feature_sets,
        candidates=candidates,
    )


def render_experiment_error(title: str, message: str) -> str:
    return "\n".join([title, "", "Errors", f"  x {message}", "", "Result: FAILED"])
