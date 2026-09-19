"""Phase-6 final refit on train+validation and the one-time test evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable

import cloudpickle
import pandas as pd

from mltool.autogluon_adapter import AutoGluonAdapter, AutoGluonError
from mltool.config import MLToolConfig
from mltool.data import DataLoadError, load_dataset
from mltool.feature_plugins import FeaturePluginError
from mltool.evaluation import EvaluationError, evaluate_predictions
from mltool.experiment import ExperimentPlan, sha256_file
from mltool.feature_materialization import (
    FeatureMaterializationError,
    _fingerprint,
    _read_manifest,
    build_feature_set_from_frames,
)
from mltool.preprocessing import PreprocessingError, preprocess_frames
from mltool.splitting import SplitError, should_stratify, split_dataset
from mltool.training import (
    TrainingError,
    _commit_staged_directory,
    _config_signature,
    convert_regression_targets,
)
from mltool.tuning import (
    TuningError,
    _read_training_artifacts,
    _validate_training_freshness,
)

FIT_SPLIT = "train_validation"


class FinalizeError(ValueError):
    """An expected Phase-6 precondition, staleness, or refit problem."""


def _stale(message: str, command: str) -> FinalizeError:
    return FinalizeError(f'{message}; run "mltool {command}" again')


@dataclass
class FinalizeInput:
    selected: dict[str, Any]
    tuning_manifest: dict[str, Any]
    prepared_manifest: dict[str, Any]


@dataclass
class FinalModelResult:
    project_root: Path
    output_path: Path
    primary_metric: str
    direction: str
    result: dict[str, Any]

    def render(self) -> str:
        result = self.result
        lines = [
            "MLTool final model",
            "",
            "Selected configuration",
            f"  {result['feature_set']} x {result['model']['name']} [{result['model']['family']}]",
            "",
            "Refit",
            f"  train+validation rows: {result['rows']['train_validation']}",
            f"  preprocessor refit: {'yes' if result['refit']['preprocessor']['enabled'] else 'disabled'}",
            f"  feature plugins refit: {', '.join(p['name'] for p in result['refit']['feature_plugins']) or 'none'}",
            f"  seed: {result['seed']}  effective_seed: {result['effective_seed']}",
            "",
            f"Test metrics ({result['rows']['test']} rows)",
        ]
        lines.extend(f"  {metric}={value:.6f}" for metric, value in result["metrics"].items())
        lines.extend(
            [
                "",
                f"Tuned validation {self.primary_metric}: "
                f"{result['selected_validation_score']:.6f}",
                f"time={result['training_seconds']:.2f}s",
                "",
                "Test split",
                "  USED (evaluated once)",
                "",
                "Result: FINALIZED",
            ]
        )
        return "\n".join(lines)


@dataclass
class PersistedFinal:
    manifest: dict[str, Any]
    result: dict[str, Any]
    warning: str | None

    def render(self) -> str:
        result = self.result
        lines = [
            "MLTool final result",
            "",
            f"Primary metric: {result['primary_metric']} ({self.manifest['metric_direction']})",
            "",
            f"Selected: {result['candidate_id']} "
            f"({result['feature_set']} x {result['model']['name']} [{result['model']['family']}])",
            f"Best hyperparameters: {json.dumps(result['best_hyperparameters'], sort_keys=True)}",
            f"seed: {result['seed']}  effective_seed: {result['effective_seed']}",
            "",
            f"Test metrics ({result['rows']['test']} rows)",
        ]
        lines.extend(f"  {metric}={value:.6f}" for metric, value in result["metrics"].items())
        lines.append(
            f"Tuned validation {result['primary_metric']}: "
            f"{result['selected_validation_score']:.6f}"
        )
        if self.warning:
            lines.extend(["", "Warnings", f"  ! {self.warning}"])
        lines.extend(["", "Test split: USED (evaluated once)"])
        return "\n".join(lines)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _dump_fitted(obj: Any, path: Path, label: str) -> None:
    """Serialize fitted state exactly like Phase 2's preprocessor.pkl (cloudpickle)."""
    try:
        with path.open("wb") as stream:
            cloudpickle.dump(obj, stream)
    except (Exception, SystemExit) as exc:
        raise FinalizeError(f"could not serialize refit {label}: {exc}") from exc


def load_finalize_input(plan: ExperimentPlan) -> FinalizeInput:
    """Refuse missing or stale tuning/training/feature/prepared artifacts."""
    config = plan.config
    root = config.config_path.parent
    tuning_path = root / ".mltool/tuning"
    selected_path = tuning_path / "selected.json"
    tuning_manifest_path = tuning_path / "manifest.json"
    if not selected_path.is_file() or not tuning_manifest_path.is_file():
        raise FinalizeError('tuning selection was not found; run "mltool tune" first')
    try:
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        tuning_manifest = json.loads(tuning_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizeError(f"tuning artifacts are unreadable: {exc}") from exc
    if not isinstance(selected, dict) or not isinstance(tuning_manifest, dict):
        raise _stale("tuning artifacts are invalid", "tune")

    # Feature artifacts were validated by build_experiment_plan. Training artifacts:
    training_manifest, _ = _read_training_artifacts(config)
    try:
        _validate_training_freshness(plan, training_manifest)
    except TuningError as exc:
        raise FinalizeError(str(exc)) from exc

    signature = {
        "task": tuning_manifest.get("task"),
        "target": tuning_manifest.get("target"),
        "primary_metric": tuning_manifest.get("primary_metric"),
        "secondary_metrics": tuning_manifest.get("secondary_metrics"),
        "feature_sets": [
            entry.get("name")
            for entry in tuning_manifest.get("feature_sets", [])
            if isinstance(entry, dict)
        ],
        "models": tuning_manifest.get("models"),
    }
    for key, value in _config_signature(config).items():
        if signature[key] != value:
            raise _stale(f'tuning artifacts are stale ("{key}" differs from the current config)', "tune")
    if config.hpo is None or tuning_manifest.get("hpo") != {
        "top_n": config.hpo.top_n,
        "num_trials": config.hpo.num_trials,
        "time_limit_seconds": config.hpo.time_limit_seconds,
        "scheduler": "local",
        "searcher": "random",
    }:
        raise _stale("tuning artifacts are stale (hpo configuration differs)", "tune")
    if tuning_manifest.get("seed") != config.training.seed:
        raise _stale("tuning artifacts are stale (training.seed differs)", "tune")
    if tuning_manifest.get("training_manifest_sha256") != sha256_file(
        root / ".mltool/training/manifest.json"
    ):
        raise _stale("tuning artifacts are stale (training was re-run after tuning)", "tune")
    if tuning_manifest.get("selected_candidate_id") != selected.get("candidate_id"):
        raise _stale("selected.json does not match the tuning manifest", "tune")

    candidate = next(
        (c for c in plan.candidates if c.candidate_id == selected.get("candidate_id")), None
    )
    if candidate is None:
        raise _stale("selected candidate is not in the current plan", "tune")
    if selected.get("model") != {
        "name": candidate.model.name,
        "family": candidate.model.family,
        "params": candidate.model.params,
    } or selected.get("feature_set") != candidate.feature_set.name:
        raise _stale("selected configuration differs from the current config", "tune")
    if selected.get("feature_artifacts") != {
        "train_sha256": candidate.feature_set.train_sha256,
        "validation_sha256": candidate.feature_set.validation_sha256,
    }:
        raise _stale("selected FeatureSet was re-materialized after tuning", "tune")
    if not isinstance(selected.get("best_hyperparameters"), dict):
        raise _stale("selected.json has no best hyperparameters", "tune")

    prepared_manifest = _validate_prepared_manifest(config)
    return FinalizeInput(
        selected=selected, tuning_manifest=tuning_manifest, prepared_manifest=prepared_manifest
    )


def _validate_prepared_manifest(config: MLToolConfig) -> dict[str, Any]:
    try:
        manifest = _read_manifest(config.config_path.parent / ".mltool/prepared/manifest.json")
        current_fingerprint = _fingerprint(config.data.path)
    except FeatureMaterializationError as exc:
        raise FinalizeError(str(exc)) from exc
    source = manifest.get("source")
    split = manifest.get("split")
    task = manifest.get("task")
    preprocessing = manifest.get("preprocessing")
    if not all(isinstance(part, dict) for part in (source, split, task, preprocessing)):
        raise _stale("prepared manifest is invalid", "prepare")
    if source.get("fingerprint") != current_fingerprint:
        raise _stale("configured source dataset changed after preparation", "prepare")
    if task != {"type": config.task.type, "target": config.task.target}:
        raise _stale("prepared task/target differs from the current config", "prepare")
    if (
        split.get("random_seed") != config.split.random_seed
        or split.get("requested_validation_ratio") != config.split.validation_ratio
        or split.get("requested_test_ratio") != config.split.test_ratio
        or split.get("stratified") != should_stratify(config.task.type, config.split.stratify)
    ):
        raise _stale("split configuration differs from the prepared split", "prepare")
    external = config.preprocessing.external
    if preprocessing.get("enabled") != external.enabled or (
        external.enabled
        and (
            preprocessing.get("entrypoint") != external.entrypoint
            or preprocessing.get("params") != external.params
        )
    ):
        raise _stale("preprocessing configuration differs from the prepared data", "prepare")
    return manifest


def _reproduce_raw_splits(config: MLToolConfig, prepared_manifest: dict[str, Any]):
    """Re-run Phase 2's split on the raw dataset and check it against the manifest."""
    try:
        dataset = load_dataset(config.data)
        splits = split_dataset(
            dataset.frame,
            target=config.task.target,
            task_type=config.task.type,
            config=config.split,
        )
    except (DataLoadError, SplitError) as exc:
        raise FinalizeError(f"could not reproduce the raw split: {exc}") from exc
    recorded = prepared_manifest["split"]
    expected_train_validation = recorded["train_rows"] + recorded["validation_rows"]
    actual_train_validation = len(splits.train) + len(splits.validation)
    if actual_train_validation != expected_train_validation or len(splits.test) != recorded[
        "test_rows"
    ]:
        raise FinalizeError(
            "reproduced raw split does not match the prepared manifest: "
            f"train+validation {actual_train_validation} vs {expected_train_validation}, "
            f'test {len(splits.test)} vs {recorded["test_rows"]}; run "mltool prepare" again'
        )
    return splits


def finalize_experiment(
    plan: ExperimentPlan,
    finalize_input: FinalizeInput,
    *,
    adapter_factory: Callable[[], AutoGluonAdapter] | None = None,
) -> FinalModelResult:
    config = plan.config
    target = config.task.target
    selected = finalize_input.selected
    candidate = next(c for c in plan.candidates if c.candidate_id == selected["candidate_id"])
    project_root = config.config_path.parent
    workspace = project_root / ".mltool"
    output_path = workspace / "final"
    started = time.perf_counter()

    splits = _reproduce_raw_splits(config, finalize_input.prepared_manifest)
    combined = pd.concat([splits.train, splits.validation])
    if combined.index.has_duplicates:
        raise FinalizeError("train and validation rows overlap after reproducing the split")
    raw_frames = {FIT_SPLIT: combined, "test": splits.test}

    # Refit the external preprocessor on train+validation only; transform both.
    try:
        preprocessed = preprocess_frames(
            raw_frames,
            fit_split=FIT_SPLIT,
            target=target,
            config=config.preprocessing.external,
            config_path=config.config_path,
        )
        spec = next(s for s in config.features.sets if s.name == candidate.feature_set.name)
        catalog = {plugin.name: plugin for plugin in config.features.plugins}
        # Only the selected FeatureSet's plugins are refit, on the same contract as Phase 3.
        materialized = build_feature_set_from_frames(
            config, preprocessed.frames, FIT_SPLIT, spec, catalog
        )
    except (PreprocessingError, FeaturePluginError, FeatureMaterializationError) as exc:
        raise FinalizeError(str(exc)) from exc

    phase3_columns = candidate.feature_set.feature_columns
    if materialized.manifest["final_feature_columns"] != phase3_columns:
        raise FinalizeError(
            "refit feature schema differs from the Phase 3 FeatureSet schema; "
            'run "mltool features" again'
        )

    fit_frame = materialized.frames[FIT_SPLIT].copy(deep=True)
    test_frame = materialized.frames["test"].copy(deep=True)
    converted = False
    if config.task.type == "regression":
        frames = {FIT_SPLIT: fit_frame, "test": test_frame}
        try:
            converted = convert_regression_targets(frames, target)
        except TrainingError as exc:
            raise FinalizeError(str(exc)) from exc
    test_target = test_frame[target].copy(deep=True)
    test_features = test_frame.drop(columns=[target])

    try:
        workspace.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".final-staging-", dir=workspace))
    except OSError as exc:
        raise FinalizeError(f"could not create final workspace: {exc}") from exc

    adapter = (adapter_factory or AutoGluonAdapter)()
    try:
        # Persist refit transform state first: a serialization failure must abort
        # before the test split is ever evaluated.
        preprocessor_artifact = None
        if preprocessed.fitted_preprocessor is not None:
            _dump_fitted(
                preprocessed.fitted_preprocessor, staging / "preprocessor.pkl", "preprocessor"
            )
            preprocessor_artifact = str(output_path / "preprocessor.pkl")
        plugin_artifacts: dict[str, str] = {}
        if materialized.fitted_plugins:
            (staging / "feature_plugins").mkdir()
        for plugin_name, fitted_plugin in materialized.fitted_plugins.items():
            _dump_fitted(
                fitted_plugin, staging / "feature_plugins" / f"{plugin_name}.pkl",
                f'feature plugin "{plugin_name}"',
            )
            plugin_artifacts[plugin_name] = str(
                output_path / "feature_plugins" / f"{plugin_name}.pkl"
            )

        # One fixed-hyperparameter fit and one test inference/evaluation, no fallback.
        try:
            output = adapter.fit_final(
                train_data=fit_frame,
                test_features=test_features,
                task=config.task,
                model=candidate.model,
                best_hyperparameters=selected["best_hyperparameters"],
                effective_seed=selected.get("effective_seed"),
                primary_metric=config.evaluation.primary_metric,
                predictor_path=staging / "predictor",
                training=config.training,
            )
            metrics = evaluate_predictions(
                task_type=config.task.type,
                evaluation=config.evaluation,
                y_true=test_target,
                predictions=output.predictions,
                probabilities=output.probabilities,
                positive_class=output.positive_class,
            )
        except (AutoGluonError, EvaluationError) as exc:
            raise FinalizeError(f"final refit failed: {exc}") from exc

        elapsed = float(time.perf_counter() - started)
        result = {
            "status": "SUCCEEDED",
            "candidate_id": candidate.candidate_id,
            "feature_set": candidate.feature_set.name,
            "model": {
                "name": candidate.model.name,
                "family": candidate.model.family,
                "params": candidate.model.params,
            },
            "task": config.task.type,
            "primary_metric": config.evaluation.primary_metric,
            "metrics": metrics,
            "selected_validation_score": selected["tuned_validation_score"],
            "best_hyperparameters": selected["best_hyperparameters"],
            "fit_hyperparameters": output.best_hyperparameters,
            "seed": config.training.seed,
            "effective_seed": selected.get("effective_seed"),
            "positive_class": output.positive_class,
            "target_conversion_applied": converted,
            "rows": {
                "train_validation": len(fit_frame),
                "test": len(test_frame),
            },
            "refit": {
                "fit_split": "train+validation",
                "preprocessor": {
                    "enabled": config.preprocessing.external.enabled,
                    "fit_rows": len(fit_frame) if config.preprocessing.external.enabled else None,
                    "resolved_entrypoint": preprocessed.resolved_entrypoint,
                    "artifact": preprocessor_artifact,
                },
                "feature_plugins": [
                    {
                        "name": plugin["name"],
                        "fit_rows": len(fit_frame),
                        "generated_columns": plugin["generated_columns"],
                        "artifact": plugin_artifacts[plugin["name"]],
                    }
                    for plugin in materialized.manifest["plugins"]
                ],
                "final_feature_columns": materialized.manifest["final_feature_columns"],
            },
            "best_model": output.best_model,
            "trained_models": output.trained_models,
            "training_seconds": elapsed,
            "predictor_path": str(output_path / "predictor"),
            "autogluon_version": output.autogluon_version,
        }
        _write_json(staging / "result.json", result)
        manifest = {
            "task": config.task.type,
            "target": target,
            "primary_metric": config.evaluation.primary_metric,
            "secondary_metrics": config.evaluation.secondary_metrics,
            "metric_direction": config.evaluation.direction,
            "autogluon_version": output.autogluon_version,
            "selected": {
                "candidate_id": candidate.candidate_id,
                "feature_set": candidate.feature_set.name,
                "model": result["model"],
                "best_hyperparameters": selected["best_hyperparameters"],
            },
            "models": [
                {"name": m.name, "family": m.family, "params": m.params} for m in config.models
            ],
            "feature_sets": [{"name": a.name} for a in plan.feature_sets],
            "seed": config.training.seed,
            "effective_seed": selected.get("effective_seed"),
            "split": {**finalize_input.prepared_manifest["split"]},
            "source_dataset_fingerprint": finalize_input.prepared_manifest["source"][
                "fingerprint"
            ],
            "tuning_manifest": str(project_root / ".mltool/tuning/manifest.json"),
            "tuning_selected": str(project_root / ".mltool/tuning/selected.json"),
            "artifacts": {
                "predictor": str(output_path / "predictor"),
                "preprocessor": preprocessor_artifact,
                "feature_plugins": plugin_artifacts,
            },
            "test_evaluations": 1,
            "test_data_used": True,
        }
        _write_json(staging / "manifest.json", manifest)
        _commit_staged_directory(staging, output_path)
    except FinalizeError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise FinalizeError(f"could not persist final artifacts: {exc}") from exc

    return FinalModelResult(
        project_root=project_root,
        output_path=output_path,
        primary_metric=config.evaluation.primary_metric,
        direction=config.evaluation.direction,
        result=result,
    )


def load_persisted_final(config: MLToolConfig) -> PersistedFinal:
    final_path = config.config_path.parent / ".mltool/final"
    manifest_path = final_path / "manifest.json"
    result_path = final_path / "result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        raise FinalizeError('final artifacts were not found; run "mltool finalize" first')
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizeError(f"final artifacts are unreadable: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(result, dict):
        raise FinalizeError('final artifacts are invalid; run "mltool finalize" again')
    signature = {
        "task": manifest.get("task"),
        "target": manifest.get("target"),
        "primary_metric": manifest.get("primary_metric"),
        "secondary_metrics": manifest.get("secondary_metrics"),
        "feature_sets": [
            entry.get("name") for entry in manifest.get("feature_sets", [])
            if isinstance(entry, dict)
        ],
        "models": manifest.get("models"),
    }
    warning = None
    if signature != _config_signature(config) or manifest.get("seed") != config.training.seed:
        warning = "current config differs from the config used to create this final model"
    else:
        try:
            current = json.loads(
                (config.config_path.parent / ".mltool/tuning/selected.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError):
            current = None
        if current is None or current.get("candidate_id") != manifest["selected"].get(
            "candidate_id"
        ) or current.get("best_hyperparameters") != manifest["selected"].get(
            "best_hyperparameters"
        ):
            warning = "the current tuning selection differs from the one used for this final model"
    return PersistedFinal(manifest=manifest, result=result, warning=warning)
