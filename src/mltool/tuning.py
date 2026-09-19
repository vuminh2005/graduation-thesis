"""Phase-5 HPO over the top Phase-4 candidates and selection of one configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable

import pandas as pd

from mltool.autogluon_adapter import (
    NO_SEARCH_SPACE_FAMILIES,
    AutoGluonAdapter,
    AutoGluonError,
    effective_model_seed,
)
from mltool.config import MLToolConfig, TrainingConfig
from mltool.evaluation import EvaluationError, evaluate_predictions
from mltool.experiment import CandidateSpec, ExperimentPlan, sha256_file
from mltool.training import (
    TrainingError,
    _autogluon_version,
    _commit_staged_directory,
    _config_signature,
    _prepare_candidate_frames,
    build_leaderboard,
)


def no_search_space_warning(family: str) -> str:
    return (
        f"HPO has no default search space for family {family}; this candidate ran "
        "with default hyperparameters, not tuned"
    )


class TuningError(ValueError):
    """An expected Phase-5 setup, staleness, or artifact problem."""


@dataclass
class TuningSelection:
    rows: list[dict[str, Any]]
    candidates: list[CandidateSpec]
    warning: str | None


@dataclass
class TuningResult:
    project_root: Path
    output_path: Path
    primary_metric: str
    direction: str
    warning: str | None
    candidate_results: list[dict[str, Any]]
    leaderboard: list[dict[str, Any]]
    selected: dict[str, Any] | None

    @property
    def successful_count(self) -> int:
        return sum(result["status"] == "SUCCEEDED" for result in self.candidate_results)

    @property
    def failed_count(self) -> int:
        return len(self.candidate_results) - self.successful_count

    @property
    def is_successful(self) -> bool:
        return self.successful_count > 0

    def render(self) -> str:
        lines = [
            "MLTool hyperparameter tuning",
            "",
            "Primary metric",
            f"  {self.primary_metric} ({self.direction})",
            "",
            "Candidates",
            f"  {len(self.candidate_results)}",
        ]
        total = len(self.candidate_results)
        for index, result in enumerate(self.candidate_results, start=1):
            lines.extend(
                [
                    "",
                    f"[{index}/{total}] {result['feature_set']} x {result['model']['name']}",
                    f"      family: {result['model']['family']}",
                    f"      {result['status']}",
                ]
            )
            if result["status"] == "SUCCEEDED":
                lines.append(f"      trained models (trials): {len(result['trained_models'])}")
                for metric, value in result["metrics"].items():
                    lines.append(f"      {metric}={value:.6f}")
                lines.append(f"      time={result['training_seconds']:.2f}s")
            else:
                lines.append(f"      error: {result['error_message']}")
        lines.extend(["", "Tuned leaderboard"])
        for row in self.leaderboard:
            if row["status"] == "SUCCEEDED":
                lines.append(
                    f"  {row['rank']}. {row['feature_set']} x {row['model']}   "
                    f"{self.primary_metric}={row['primary_score']:.6f}"
                )
        warnings = []
        if self.warning:
            warnings.append(self.warning)
        for result in self.candidate_results:
            if result.get("hpo_warning"):
                warnings.append(f"{result['candidate_id']}: {result['hpo_warning']}")
        if self.failed_count:
            warnings.append(f"{self.failed_count} candidate(s) failed")
        if warnings:
            lines.extend(["", "Warnings", *[f"  ! {warning}" for warning in warnings]])
        lines.extend(["", "Selected configuration"])
        if self.selected is None:
            lines.append("  none")
        else:
            lines.append(
                f"  {self.selected['feature_set']} x {self.selected['model']['name']} "
                f"[{self.selected['model']['family']}] "
                f"{self.primary_metric}={self.selected['tuned_validation_score']:.6f}"
            )
        lines.extend(
            [
                "",
                "Test split",
                "  NOT USED",
                "",
                f'Result: {"TUNED" if self.is_successful else "FAILED"}',
            ]
        )
        return "\n".join(lines)


@dataclass
class PersistedTuning:
    manifest: dict[str, Any]
    rows: list[dict[str, Any]]
    warning: str | None

    def render(self) -> str:
        lines = [
            "MLTool tuning leaderboard",
            "",
            f"Primary metric: {self.manifest['primary_metric']} "
            f"({self.manifest['metric_direction']})",
            "",
            "Rank  FeatureSet  Model  Family  Score  Time  Status",
        ]
        for row in self.rows:
            rank = str(row["rank"]) if row["rank"] is not None else "-"
            score = (
                f"{row['primary_score']:.6f}" if row.get("primary_score") is not None else "-"
            )
            seconds = (
                f"{row['training_seconds']:.2f}s"
                if row.get("training_seconds") is not None
                else "-"
            )
            lines.append(
                f"{rank:<5} {row['feature_set']:<11} {row['model']:<16} "
                f"{row['family']:<6} {score:<10} {seconds:<8} {row['status']}"
            )
        selected = self.manifest.get("selected_candidate_id")
        lines.extend(["", f"Selected: {selected if selected else 'none'}"])
        if self.warning:
            lines.extend(["", "Warnings", f"  ! {self.warning}"])
        lines.extend(["", "Test split: NOT USED"])
        return "\n".join(lines)


def _read_training_artifacts(config: MLToolConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training_path = config.config_path.parent / ".mltool/training"
    manifest_path = training_path / "manifest.json"
    leaderboard_path = training_path / "leaderboard.json"
    if not manifest_path.is_file() or not leaderboard_path.is_file():
        raise TrainingError('training artifacts were not found; run "mltool train" first')
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"training leaderboard artifacts are unreadable: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(rows, list):
        raise TrainingError('training leaderboard artifacts are invalid; run "mltool train" again')
    return manifest, rows


def _training_stale(message: str) -> TuningError:
    return TuningError(
        f'training artifacts are stale ({message}); run "mltool train" again'
    )


def _validate_training_freshness(plan: ExperimentPlan, manifest: dict[str, Any]) -> None:
    config = plan.config
    expected = _config_signature(config)
    actual = {
        "task": manifest.get("task"),
        "target": manifest.get("target"),
        "primary_metric": manifest.get("primary_metric"),
        "secondary_metrics": manifest.get("secondary_metrics"),
        "feature_sets": [
            entry.get("name")
            for entry in manifest.get("feature_sets", [])
            if isinstance(entry, dict)
        ],
        "models": manifest.get("models"),
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise _training_stale(f'"{key}" differs from the current config')
    if manifest.get("source_dataset_fingerprint") != plan.feature_manifest.get(
        "source_dataset_fingerprint"
    ):
        raise _training_stale("source dataset fingerprint differs from the feature artifacts")
    trained_hashes = {
        entry.get("name"): (entry.get("train_sha256"), entry.get("validation_sha256"))
        for entry in manifest.get("feature_sets", [])
        if isinstance(entry, dict)
    }
    for artifact in plan.feature_sets:
        if trained_hashes.get(artifact.name) != (
            artifact.train_sha256,
            artifact.validation_sha256,
        ):
            raise _training_stale(
                f'feature set "{artifact.name}" was re-materialized after training'
            )


def select_top_rows(
    rows: list[dict[str, Any]], *, top_n: int, direction: str
) -> tuple[list[dict[str, Any]], str | None]:
    """Pick the best ``top_n`` SUCCEEDED rows by primary score, stable on ties."""
    succeeded = [
        row
        for row in rows
        if row.get("status") == "SUCCEEDED" and row.get("primary_score") is not None
    ]
    succeeded.sort(key=lambda row: row["primary_score"], reverse=direction == "maximize")
    selected = succeeded[:top_n]
    warning = None
    if len(selected) < top_n:
        warning = (
            f"only {len(selected)} succeeded Phase-4 candidate(s) available; "
            f"hpo.top_n is {top_n}"
        )
    return selected, warning


def build_tuning_selection(plan: ExperimentPlan) -> TuningSelection:
    config = plan.config
    if config.hpo is None:
        raise TuningError('no "hpo" section is configured; add it to mltool.yaml')
    manifest, rows = _read_training_artifacts(config)
    _validate_training_freshness(plan, manifest)
    selected_rows, warning = select_top_rows(
        rows, top_n=config.hpo.top_n, direction=config.evaluation.direction
    )
    if not selected_rows:
        raise TuningError(
            'no succeeded Phase-4 candidates are available for tuning; run "mltool train" again'
        )
    by_id = {candidate.candidate_id: candidate for candidate in plan.candidates}
    candidates: list[CandidateSpec] = []
    for row in selected_rows:
        candidate = by_id.get(row.get("candidate_id"))
        if candidate is None:
            raise _training_stale(f'candidate "{row.get("candidate_id")}" is not in the plan')
        candidates.append(candidate)
    return TuningSelection(rows=selected_rows, candidates=candidates, warning=warning)


def _tune_candidate(
    plan: ExperimentPlan,
    candidate: CandidateSpec,
    phase4_score: float,
    *,
    staging_candidate_path: Path,
    final_candidate_path: Path,
    adapter: AutoGluonAdapter,
) -> dict[str, Any]:
    assert plan.config.hpo is not None
    started = time.perf_counter()
    train, validation_features, validation_target, converted = _prepare_candidate_frames(
        plan, candidate
    )
    output = adapter.fit_predict(
        train_data=train,
        validation_features=validation_features,
        task=plan.config.task,
        model=candidate.model,
        primary_metric=plan.config.evaluation.primary_metric,
        predictor_path=staging_candidate_path / "predictor",
        training=plan.config.training,
        hpo=plan.config.hpo,
    )
    metrics = evaluate_predictions(
        task_type=plan.config.task.type,
        evaluation=plan.config.evaluation,
        y_true=validation_target,
        predictions=output.predictions,
        probabilities=output.probabilities,
        positive_class=output.positive_class,
    )
    return {
        "candidate_id": candidate.candidate_id,
        "feature_set": candidate.feature_set.name,
        "feature_artifacts": {
            "train_sha256": candidate.feature_set.train_sha256,
            "validation_sha256": candidate.feature_set.validation_sha256,
        },
        "model": {
            "name": candidate.model.name,
            "family": candidate.model.family,
            "params": candidate.model.params,
        },
        "task": plan.config.task.type,
        "primary_metric": plan.config.evaluation.primary_metric,
        "metrics": metrics,
        "phase4_primary_score": phase4_score,
        "positive_class": output.positive_class,
        "target_conversion_applied": converted,
        "training_seconds": float(time.perf_counter() - started),
        "hpo": {
            "num_trials": plan.config.hpo.num_trials,
            "time_limit_seconds": plan.config.hpo.time_limit_seconds,
            "scheduler": "local",
            "searcher": "random",
        },
        "seed": plan.config.training.seed,
        "effective_seed": effective_model_seed(candidate.model, plan.config.training),
        "hpo_effective": candidate.model.family not in NO_SEARCH_SPACE_FAMILIES,
        "hpo_warning": (
            no_search_space_warning(candidate.model.family)
            if candidate.model.family in NO_SEARCH_SPACE_FAMILIES
            else None
        ),
        "best_model": output.best_model,
        "best_hyperparameters": output.best_hyperparameters,
        "predictor_path": str(final_candidate_path / "predictor"),
        "autogluon_version": output.autogluon_version,
        "trained_models": output.trained_models,
        "status": "SUCCEEDED",
    }


def _failure(candidate: CandidateSpec, message: str, training: TrainingConfig) -> dict[str, Any]:
    return {
        "seed": training.seed,
        "effective_seed": effective_model_seed(candidate.model, training),
        "candidate_id": candidate.candidate_id,
        "feature_set": candidate.feature_set.name,
        "feature_artifacts": {
            "train_sha256": candidate.feature_set.train_sha256,
            "validation_sha256": candidate.feature_set.validation_sha256,
        },
        "model": {
            "name": candidate.model.name,
            "family": candidate.model.family,
            "params": candidate.model.params,
        },
        "status": "FAILED",
        "error_message": message,
    }


def build_selected_configuration(
    leaderboard: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
    *,
    primary_metric: str,
) -> dict[str, Any] | None:
    """The best tuned candidate, i.e. rank 1 of the tuned leaderboard."""
    ranked = [row for row in leaderboard if row["status"] == "SUCCEEDED"]
    if not ranked:
        return None
    best_id = ranked[0]["candidate_id"]
    result = next(item for item in candidate_results if item["candidate_id"] == best_id)
    return {
        "candidate_id": result["candidate_id"],
        "feature_set": result["feature_set"],
        "model": result["model"],
        "best_hyperparameters": result["best_hyperparameters"],
        "best_model": result["best_model"],
        "hpo_effective": result["hpo_effective"],
        "hpo_warning": result["hpo_warning"],
        "seed": result["seed"],
        "effective_seed": result["effective_seed"],
        "primary_metric": primary_metric,
        "tuned_validation_score": result["metrics"][primary_metric],
        "phase4_primary_score": result["phase4_primary_score"],
        "validation_metrics": result["metrics"],
        "feature_artifacts": result["feature_artifacts"],
        "positive_class": result["positive_class"],
        "autogluon_version": result["autogluon_version"],
        "predictor_path": result["predictor_path"],
        "test_data_used": False,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def tune_experiment(
    plan: ExperimentPlan,
    selection: TuningSelection,
    *,
    adapter_factory: Callable[[], AutoGluonAdapter] | None = None,
) -> TuningResult:
    config = plan.config
    assert config.hpo is not None
    project_root = config.config_path.parent
    workspace = project_root / ".mltool"
    output_path = workspace / "tuning"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".tuning-staging-", dir=workspace))
        (staging / "candidates").mkdir()
    except OSError as exc:
        raise TuningError(f"could not create tuning workspace: {exc}") from exc

    adapter_factory = adapter_factory or AutoGluonAdapter
    candidate_results: list[dict[str, Any]] = []
    try:
        # Strictly sequential: one candidate (and one local trial) at a time.
        for candidate, row in zip(selection.candidates, selection.rows, strict=True):
            staging_candidate = staging / "candidates" / candidate.candidate_id
            final_candidate = output_path / "candidates" / candidate.candidate_id
            staging_candidate.mkdir()
            try:
                result = _tune_candidate(
                    plan,
                    candidate,
                    row["primary_score"],
                    staging_candidate_path=staging_candidate,
                    final_candidate_path=final_candidate,
                    adapter=adapter_factory(),
                )
            except (TrainingError, AutoGluonError, EvaluationError) as exc:
                result = _failure(candidate, str(exc), config.training)
            _write_json(staging_candidate / "result.json", result)
            candidate_results.append(result)

        leaderboard = build_leaderboard(
            candidate_results,
            primary_metric=config.evaluation.primary_metric,
            direction=config.evaluation.direction,
        )
        _write_json(staging / "leaderboard.json", leaderboard)
        pd.DataFrame(leaderboard).to_csv(staging / "leaderboard.csv", index=False)
        selected = build_selected_configuration(
            leaderboard, candidate_results, primary_metric=config.evaluation.primary_metric
        )
        if selected is not None:
            _write_json(staging / "selected.json", selected)
        succeeded = sum(result["status"] == "SUCCEEDED" for result in candidate_results)
        manifest = {
            "task": config.task.type,
            "target": config.task.target,
            "primary_metric": config.evaluation.primary_metric,
            "secondary_metrics": config.evaluation.secondary_metrics,
            "metric_direction": config.evaluation.direction,
            "autogluon_version": next(
                (
                    result["autogluon_version"]
                    for result in candidate_results
                    if result["status"] == "SUCCEEDED"
                ),
                _autogluon_version(),
            ),
            "hpo": {
                "top_n": config.hpo.top_n,
                "num_trials": config.hpo.num_trials,
                "time_limit_seconds": config.hpo.time_limit_seconds,
                "scheduler": "local",
                "searcher": "random",
            },
            "seed": config.training.seed,
            "effective_seed": {
                candidate.candidate_id: effective_model_seed(candidate.model, config.training)
                for candidate in selection.candidates
            },
            "training_manifest": str(project_root / ".mltool/training/manifest.json"),
            "training_manifest_sha256": sha256_file(
                project_root / ".mltool/training/manifest.json"
            ),
            "source_dataset_fingerprint": plan.feature_manifest.get("source_dataset_fingerprint"),
            "models": [
                {"name": model.name, "family": model.family, "params": model.params}
                for model in config.models
            ],
            "feature_sets": [
                {"name": artifact.name}
                for artifact in plan.feature_sets
            ],
            "candidate_ids": [candidate.candidate_id for candidate in selection.candidates],
            "selection_warning": selection.warning,
            "successful_count": succeeded,
            "failed_count": len(candidate_results) - succeeded,
            "selected_candidate_id": selected["candidate_id"] if selected else None,
            "leaderboard": {
                "csv": str(output_path / "leaderboard.csv"),
                "json": str(output_path / "leaderboard.json"),
            },
            "selected": str(output_path / "selected.json") if selected else None,
            "test_data_used": False,
        }
        _write_json(staging / "manifest.json", manifest)
        _commit_staged_directory(staging, output_path)
    except Exception as exc:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise TuningError(f"could not persist tuning artifacts: {exc}") from exc

    return TuningResult(
        project_root=project_root,
        output_path=output_path,
        primary_metric=config.evaluation.primary_metric,
        direction=config.evaluation.direction,
        warning=selection.warning,
        candidate_results=candidate_results,
        leaderboard=leaderboard,
        selected=selected,
    )


def load_persisted_tuning(config: MLToolConfig) -> PersistedTuning:
    tuning_path = config.config_path.parent / ".mltool/tuning"
    manifest_path = tuning_path / "manifest.json"
    leaderboard_path = tuning_path / "leaderboard.json"
    if not manifest_path.is_file() or not leaderboard_path.is_file():
        raise TuningError('tuning artifacts were not found; run "mltool tune" first')
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TuningError(f"tuning leaderboard artifacts are unreadable: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(rows, list):
        raise TuningError('tuning leaderboard artifacts are invalid; run "mltool tune" again')
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
    expected = _config_signature(config)
    hpo_matches = config.hpo is not None and manifest.get("hpo") == {
        "top_n": config.hpo.top_n,
        "num_trials": config.hpo.num_trials,
        "time_limit_seconds": config.hpo.time_limit_seconds,
        "scheduler": "local",
        "searcher": "random",
    }
    warning = None
    if signature != expected or not hpo_matches:
        warning = "current config differs from the config used to create this tuning leaderboard"
    return PersistedTuning(manifest=manifest, rows=rows, warning=warning)
