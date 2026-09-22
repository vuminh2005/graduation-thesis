"""Sequential Phase-4 candidate training and persisted global leaderboard."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable
from uuid import uuid4

import pandas as pd
from pandas.api.types import is_numeric_dtype

from mltool.autogluon_adapter import AutoGluonAdapter, AutoGluonError, effective_model_seed
from mltool.config import MLToolConfig, TrainingConfig
from mltool.evaluation import EvaluationError, evaluate_predictions
from mltool.experiment import CandidateSpec, ExperimentPlan, ExperimentError


class TrainingError(ValueError):
    """An expected Phase-4 training artifact/setup problem."""


@dataclass
class TrainingResult:
    project_root: Path
    output_path: Path
    primary_metric: str
    direction: str
    candidate_results: list[dict[str, Any]]
    leaderboard: list[dict[str, Any]]
    cv: dict[str, Any] | None = None

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
        task = "candidate training"
        lines = [
            "MLTool candidate training",
            "",
            "Primary metric",
            f"  {self.primary_metric} ({self.direction})",
            "",
            "Candidates",
            f"  {len(self.candidate_results)}",
        ]
        lines.extend(_cv_header(self.cv))
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
                cv = result.get("cv")
                std = cv["metric_std"] if cv else None
                folds = cv["total_fits_per_candidate"] if cv else None
                for metric, value in result["metrics"].items():
                    lines.append(
                        f"      {metric}="
                        f"{format_score(value, std[metric] if std else None, folds)}"
                    )
                lines.append(f"      time={result['training_seconds']:.2f}s")
            else:
                lines.append(f"      error: {result['error_message']}")
        lines.extend(["", "Global leaderboard"])
        succeeded = [row for row in self.leaderboard if row["status"] == "SUCCEEDED"]
        for row in succeeded:
            score = format_score(row["primary_score"], *_row_score_parts(row))
            lines.append(
                f"  {row['rank']}. {row['feature_set']} x {row['model']}   "
                f"{self.primary_metric}={score}"
            )
        if self.failed_count:
            lines.extend(["", f"Warnings", f"  ! {self.failed_count} candidate(s) failed"])
        lines.extend(
            [
                "",
                "Test split",
                "  NOT USED",
                "",
                f'Result: {"TRAINED" if self.is_successful else "FAILED"}',
            ]
        )
        return "\n".join(lines)


@dataclass
class PersistedLeaderboard:
    manifest: dict[str, Any]
    rows: list[dict[str, Any]]
    warning: str | None

    def render(self) -> str:
        metric = self.manifest["primary_metric"]
        lines = [
            "MLTool global leaderboard",
            "",
            f"Primary metric: {metric} ({self.manifest['metric_direction']})",
        ]
        lines.extend(_cv_header(self.manifest.get("cv")))
        lines.extend(["", "Rank  FeatureSet  Model  Family  Score  Time  Status"])
        for row in self.rows:
            rank = str(row["rank"]) if row["rank"] is not None else "-"
            score = (
                format_score(row["primary_score"], *_row_score_parts(row))
                if row.get("primary_score") is not None
                else "-"
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
        if self.warning:
            lines.extend(["", "Warnings", f"  ! {self.warning}"])
        lines.extend(["", "Test split: NOT USED"])
        return "\n".join(lines)


def format_score(value: float, std: float | None, folds: int | None) -> str:
    """``0.912345`` for a holdout score, ``0.912345 +/- 0.021 (15 folds)`` for a CV mean."""
    if std is None or folds is None:
        return f"{value:.6f}"
    return f"{value:.6f} +/- {std:.6f} ({folds} folds)"


def _row_score_parts(row: dict[str, Any]) -> tuple[float | None, int | None]:
    return row.get("primary_score_std"), row.get("folds")


def _cv_header(cv: dict[str, Any] | None) -> list[str]:
    if not cv:
        return []
    return [
        "",
        "Cross-validation",
        f"  {cv['folds']} folds x {cv['repeats']} repeat(s) = "
        f"{cv['total_fits_per_candidate']} fits per candidate"
        f"{' (stratified)' if cv.get('stratified') else ''}",
        f"  development rows: {cv['development_rows']}  |  test rows: not used",
        f"  fold fingerprint: {cv['fold_fingerprint'][:16]}",
    ]


def _autogluon_version() -> str:
    try:
        return version("autogluon.tabular")
    except PackageNotFoundError:  # pragma: no cover - broken environment
        return "unknown"


def convert_regression_targets(frames: dict[str, pd.DataFrame], target: str) -> bool:
    """Strictly convert regression targets to numeric in place; True if any changed dtype."""
    applied = not all(is_numeric_dtype(frame[target]) for frame in frames.values())
    try:
        for frame in frames.values():
            frame[target] = pd.to_numeric(frame[target], errors="raise")
    except (TypeError, ValueError) as exc:
        raise TrainingError(
            f'regression target "{target}" cannot be converted strictly to numeric data: {exc}'
        ) from exc
    return applied


def _prepare_candidate_frames(
    plan: ExperimentPlan,
    candidate: CandidateSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, bool]:
    try:
        train = pd.read_parquet(candidate.feature_set.train_path, engine="pyarrow")
        validation = pd.read_parquet(
            candidate.feature_set.validation_path, engine="pyarrow"
        )
    except Exception as exc:
        raise TrainingError(f"could not read candidate feature artifacts: {exc}") from exc
    target = plan.config.task.target
    train = train.copy(deep=True)
    validation = validation.copy(deep=True)
    conversion_applied = False
    if plan.config.task.type == "regression":
        frames = {"train": train, "validation": validation}
        conversion_applied = convert_regression_targets(frames, target)
        train, validation = frames["train"], frames["validation"]
    validation_target = validation[target].copy(deep=True)
    validation_features = validation.drop(columns=[target]).copy(deep=True)
    return train, validation_features, validation_target, conversion_applied


def _candidate_failure(
    candidate: CandidateSpec, message: str, training: TrainingConfig
) -> dict[str, Any]:
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


def _cv_train_candidate(
    plan: ExperimentPlan,
    candidate: CandidateSpec,
    cv_plan: Any,
    *,
    adapter_factory: Callable[[], AutoGluonAdapter],
) -> dict[str, Any]:
    """Cross-validated scoring of one candidate; every fold refits all state."""
    from mltool.cross_validation import cv_score_candidate

    config = plan.config
    started = time.perf_counter()
    spec = next(s for s in config.features.sets if s.name == candidate.feature_set.name)
    catalog = {plugin.name: plugin for plugin in config.features.plugins}
    score = cv_score_candidate(
        config,
        cv_plan,
        spec,
        candidate.model,
        catalog,
        adapter_factory=adapter_factory,
        label=candidate.candidate_id,
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
        "task": config.task.type,
        "primary_metric": config.evaluation.primary_metric,
        "metrics": score.metrics,
        "cv": score.as_record(cv_plan),
        "seed": config.training.seed,
        "effective_seed": effective_model_seed(candidate.model, config.training),
        "positive_class": score.positive_class,
        "target_conversion_applied": score.target_conversion_applied,
        "training_seconds": float(time.perf_counter() - started),
        # Each fold has its own model, and nothing downstream loads a training
        # predictor, so fold predictors are scratch and never persisted.
        "predictor_path": None,
        "predictor_persisted": False,
        "autogluon_version": score.autogluon_version,
        "trained_models": score.trained_models,
        "status": "SUCCEEDED",
    }


def _train_candidate(
    plan: ExperimentPlan,
    candidate: CandidateSpec,
    *,
    staging_candidate_path: Path,
    final_candidate_path: Path,
    adapter: AutoGluonAdapter,
) -> dict[str, Any]:
    started = time.perf_counter()
    train, validation_features, validation_target, converted = _prepare_candidate_frames(
        plan, candidate
    )
    predictor_staging = staging_candidate_path / "predictor"
    output = adapter.fit_predict(
        train_data=train,
        validation_features=validation_features,
        task=plan.config.task,
        model=candidate.model,
        primary_metric=plan.config.evaluation.primary_metric,
        predictor_path=predictor_staging,
        training=plan.config.training,
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
        "seed": plan.config.training.seed,
        "effective_seed": effective_model_seed(candidate.model, plan.config.training),
        "positive_class": output.positive_class,
        "target_conversion_applied": converted,
        "training_seconds": float(time.perf_counter() - started),
        "predictor_path": str(final_candidate_path / "predictor"),
        "autogluon_version": output.autogluon_version,
        "trained_models": output.trained_models,
        "status": "SUCCEEDED",
    }


def build_leaderboard(
    candidate_results: list[dict[str, Any]],
    *,
    primary_metric: str,
    direction: str,
) -> list[dict[str, Any]]:
    successes = [result for result in candidate_results if result["status"] == "SUCCEEDED"]
    failures = [result for result in candidate_results if result["status"] == "FAILED"]
    successes.sort(
        key=lambda result: result["metrics"][primary_metric],
        reverse=direction == "maximize",
    )
    rows: list[dict[str, Any]] = []
    for rank, result in enumerate(successes, start=1):
        row = {
            "rank": rank,
            "candidate_id": result["candidate_id"],
            "feature_set": result["feature_set"],
            "model": result["model"]["name"],
            "family": result["model"]["family"],
            "primary_score": result["metrics"][primary_metric],
            "training_seconds": result["training_seconds"],
            "status": "SUCCEEDED",
        }
        cv = result.get("cv")
        if cv:
            row["primary_score_std"] = cv["metric_std"][primary_metric]
            row["folds"] = cv["total_fits_per_candidate"]
        row.update(result["metrics"])
        rows.append(row)
    for result in failures:
        rows.append(
            {
                "rank": None,
                "candidate_id": result["candidate_id"],
                "feature_set": result["feature_set"],
                "model": result["model"]["name"],
                "family": result["model"]["family"],
                "primary_score": None,
                "training_seconds": None,
                "status": "FAILED",
                "error_message": result["error_message"],
            }
        )
    return rows


def _commit_staged_directory(staging: Path, output_path: Path) -> None:
    backup: Path | None = None
    if output_path.exists():
        backup = output_path.parent / f".training-backup-{uuid4().hex}"
        output_path.rename(backup)
    try:
        staging.rename(output_path)
    except Exception:
        if backup is not None and backup.exists() and not output_path.exists():
            backup.rename(output_path)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def train_experiment(
    plan: ExperimentPlan,
    *,
    adapter_factory: Callable[[], AutoGluonAdapter] | None = None,
) -> TrainingResult:
    project_root = plan.config.config_path.parent
    workspace = project_root / ".mltool"
    output_path = workspace / "training"
    try:
        workspace.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".training-staging-", dir=workspace))
        (staging / "candidates").mkdir()
    except OSError as exc:
        raise TrainingError(f"could not create training workspace: {exc}") from exc

    adapter_factory = adapter_factory or AutoGluonAdapter
    cv_plan = None
    if plan.config.evaluation.cv is not None:
        from mltool.cross_validation import build_cv_plan, cost_warning, progress

        try:
            cv_plan = build_cv_plan(plan.config)
        except Exception as exc:
            shutil.rmtree(staging, ignore_errors=True)
            raise TrainingError(str(exc)) from exc
        progress(cost_warning(len(plan.candidates), cv_plan, "train"))
    candidate_results: list[dict[str, Any]] = []
    try:
        for candidate in plan.candidates:
            staging_candidate = staging / "candidates" / candidate.candidate_id
            final_candidate = output_path / "candidates" / candidate.candidate_id
            staging_candidate.mkdir()
            try:
                if cv_plan is not None:
                    result = _cv_train_candidate(
                        plan, candidate, cv_plan, adapter_factory=adapter_factory
                    )
                else:
                    result = _train_candidate(
                        plan,
                        candidate,
                        staging_candidate_path=staging_candidate,
                        final_candidate_path=final_candidate,
                        adapter=adapter_factory(),
                    )
            except (TrainingError, AutoGluonError, EvaluationError) as exc:
                # Candidate-specific failures are data/model failures, not matrix
                # failures. Continue sequentially and persist a concise result.
                result = _candidate_failure(candidate, str(exc), plan.config.training)
            (staging_candidate / "result.json").write_text(
                json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            candidate_results.append(result)

        leaderboard = build_leaderboard(
            candidate_results,
            primary_metric=plan.config.evaluation.primary_metric,
            direction=plan.config.evaluation.direction,
        )
        (staging / "leaderboard.json").write_text(
            json.dumps(leaderboard, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(leaderboard).to_csv(staging / "leaderboard.csv", index=False)
        succeeded = sum(result["status"] == "SUCCEEDED" for result in candidate_results)
        manifest = {
            "task": plan.config.task.type,
            "target": plan.config.task.target,
            "primary_metric": plan.config.evaluation.primary_metric,
            "secondary_metrics": plan.config.evaluation.secondary_metrics,
            "metric_direction": plan.config.evaluation.direction,
            "autogluon_version": next(
                (
                    result["autogluon_version"]
                    for result in candidate_results
                    if result["status"] == "SUCCEEDED"
                ),
                _autogluon_version(),
            ),
            "feature_manifest": str(plan.feature_manifest_path),
            "source_dataset_fingerprint": plan.feature_manifest.get(
                "source_dataset_fingerprint"
            ),
            "feature_sets": [
                {
                    "name": artifact.name,
                    "manifest": str(artifact.manifest_path),
                    "source_columns": artifact.manifest.get("source_columns"),
                    "plugins": artifact.manifest.get("plugins"),
                    "train_sha256": artifact.train_sha256,
                    "validation_sha256": artifact.validation_sha256,
                }
                for artifact in plan.feature_sets
            ],
            "models": [
                {"name": model.name, "family": model.family, "params": model.params}
                for model in plan.config.models
            ],
            "candidate_ids": [candidate.candidate_id for candidate in plan.candidates],
            **({"cv": cv_plan.summary()} if cv_plan is not None else {}),
            "seed": plan.config.training.seed,
            "effective_seed": {
                candidate.candidate_id: effective_model_seed(
                    candidate.model, plan.config.training
                )
                for candidate in plan.candidates
            },
            "successful_count": succeeded,
            "failed_count": len(candidate_results) - succeeded,
            "leaderboard": {
                "csv": str(output_path / "leaderboard.csv"),
                "json": str(output_path / "leaderboard.json"),
            },
            "test_data_used": False,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        _commit_staged_directory(staging, output_path)
    except Exception as exc:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if isinstance(exc, TrainingError):
            raise
        raise TrainingError(f"could not persist training artifacts: {exc}") from exc

    return TrainingResult(
        project_root=project_root,
        output_path=output_path,
        primary_metric=plan.config.evaluation.primary_metric,
        direction=plan.config.evaluation.direction,
        candidate_results=candidate_results,
        leaderboard=leaderboard,
        cv=cv_plan.summary() if cv_plan is not None else None,
    )


def cv_signature(config: MLToolConfig) -> dict[str, int] | None:
    """The cross-validation settings as they appear in a config signature."""
    cv = config.evaluation.cv
    return None if cv is None else {"folds": cv.folds, "repeats": cv.repeats}


def _config_signature(config: MLToolConfig) -> dict[str, Any]:
    return {
        "task": config.task.type,
        "target": config.task.target,
        "primary_metric": config.evaluation.primary_metric,
        "secondary_metrics": config.evaluation.secondary_metrics,
        "cv": cv_signature(config),
        "feature_sets": [feature_set.name for feature_set in config.features.sets],
        "models": [
            {"name": model.name, "family": model.family, "params": model.params}
            for model in config.models
        ],
    }


def _manifest_cv_signature(manifest: dict[str, Any]) -> dict[str, int] | None:
    """A manifest written before Phase 9 has no cv block, which means holdout."""
    cv = manifest.get("cv")
    if not isinstance(cv, dict):
        return None
    return {"folds": cv["folds"], "repeats": cv["repeats"]}


def load_persisted_leaderboard(config: MLToolConfig) -> PersistedLeaderboard:
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
    training_signature = {
        "task": manifest.get("task"),
        "target": manifest.get("target"),
        "primary_metric": manifest.get("primary_metric"),
        "secondary_metrics": manifest.get("secondary_metrics"),
        "cv": _manifest_cv_signature(manifest),
        "feature_sets": [
            entry.get("name") for entry in manifest.get("feature_sets", [])
            if isinstance(entry, dict)
        ],
        "models": manifest.get("models"),
    }
    warning = None
    if training_signature != _config_signature(config):
        warning = "current config differs from the config used to create this leaderboard"
    return PersistedLeaderboard(manifest=manifest, rows=rows, warning=warning)
