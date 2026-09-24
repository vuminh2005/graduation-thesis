"""Phase-7 best-effort MLflow tracking on a project-local file store.

Tracking never gates a command: every public function catches its own failures
and returns a warning string instead of raising. MLflow's Model Registry is
deliberately not used (see ``registry.py`` for MLTool's own local registry).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from typing import Any

from mltool.config import MLToolConfig

# MLflow >= 3.7 refuses the file store unless it is explicitly allowed. MLTool's
# tracking is intentionally local and file based, so opt in (a user's own value wins).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

MAX_PARAM_LENGTH = 500


def tracking_uri(project_root: Path) -> str:
    return (project_root / ".mltool" / "mlflow").resolve().as_uri()


def experiment_name(config: MLToolConfig) -> str:
    return config.project.name.strip() or config.config_path.parent.name or "mltool"


def _param(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= MAX_PARAM_LENGTH else text[: MAX_PARAM_LENGTH - 3] + "..."


def _flatten(prefix: str, values: dict[str, Any] | None) -> dict[str, Any]:
    return {f"{prefix}.{key}": value for key, value in (values or {}).items()}


def _custom_model_params(result: dict[str, Any]) -> dict[str, Any]:
    """A SKLEARN model's entrypoint, input mode, source hash and seed note."""
    model = result.get("model") or {}
    params = {f"model_{key}": model[key] for key in ("entrypoint", "input", "source_sha256") if key in model}
    if "seed_note" in result:
        params["seed_note"] = result["seed_note"]
    return params


def search_space_text(spec: dict[str, Any]) -> str:
    """``real[0.005,0.2,log]``, ``int[16,128,default=31]``, ``categorical[true,false]``."""
    if spec["type"] == "categorical":
        parts = [json.dumps(value) for value in spec["values"]]
    else:
        parts = [json.dumps(spec["low"]), json.dumps(spec["high"])]
        if spec.get("log"):
            parts.append("log")
    if spec.get("default") is not None and spec["type"] != "categorical":
        parts.append(f"default={json.dumps(spec['default'])}")
    return f"{spec['type']}[{','.join(parts)}]"


def _search_space_params(result: dict[str, Any]) -> dict[str, str]:
    """``ss.<key>`` for each declared range, ``ss_default.<key>`` for each of
    AutoGluon's default ranges that the merge left searched."""
    params = {f"ss.{key}": search_space_text(spec)
              for key, spec in (result.get("search_space") or {}).items()}
    for key, spec in (result.get("effective_search_space") or {}).items():
        if spec.get("source") == "autogluon_default":
            params[f"ss_default.{key}"] = search_space_text(spec)
    return params


def _cv_metrics(result: dict[str, Any]) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    """Std metrics, the per-fold series and cv params for a cross-validated result."""
    cv = result.get("cv")
    if not cv:
        return {}, [], {}
    std = {f"{name}_std": value for name, value in cv.get("metric_std", {}).items()}
    params = {
        "cv_folds": cv.get("folds"),
        "cv_repeats": cv.get("repeats"),
        "cv_fits": cv.get("total_fits_per_candidate"),
        "cv_fold_fingerprint": cv.get("fold_fingerprint"),
    }
    return std, list(cv.get("fold_metrics", [])), params


class _Tracker:
    def __init__(self, config: MLToolConfig) -> None:
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        from mlflow import MlflowClient

        self.client = MlflowClient(tracking_uri=tracking_uri(config.config_path.parent))
        name = experiment_name(config)
        experiment = self.client.get_experiment_by_name(name)
        self.experiment_id = (
            experiment.experiment_id if experiment else self.client.create_experiment(name)
        )

    def log_run(
        self,
        *,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, Any],
        failed: bool = False,
        fold_metrics: list[dict[str, Any]] | None = None,
    ) -> str:
        from mlflow.entities import Metric, Param, RunTag

        run = self.client.create_run(self.experiment_id, run_name=run_name)
        run_id = run.info.run_id
        now = int(time.time() * 1000)
        series = [Metric(key, float(value), now, 0) for key, value in metrics.items()]
        # Per-fold values share the mean's metric name, stepped by fold index, so
        # MLflow plots them as a series under the aggregate.
        for step, entry in enumerate(fold_metrics or []):
            series.extend(
                Metric(f"fold_{key}", float(value), now, step)
                for key, value in entry["metrics"].items()
            )
        self.client.log_batch(
            run_id,
            metrics=series,
            params=[Param(key, _param(value)) for key, value in params.items()],
            tags=[RunTag(key, _param(value)) for key, value in tags.items()],
        )
        self.client.set_terminated(run_id, status="FAILED" if failed else "FINISHED")
        return run_id


def _safely(config: MLToolConfig, log: Any) -> tuple[list[str], str | None]:
    try:
        return log(_Tracker(config)), None
    except Exception as exc:  # noqa: BLE001 - tracking must never abort a command
        return [], f"MLflow tracking failed and was skipped: {exc}"


def log_training(config: MLToolConfig, result_set: Any):
    """One run per Phase-4 candidate (successful or failed)."""

    def log(tracker: _Tracker) -> list[str]:
        run_ids = []
        for result in result_set.candidate_results:
            failed = result["status"] != "SUCCEEDED"
            params = {
                "feature_set": result["feature_set"],
                "model_name": result["model"]["name"],
                "model_family": result["model"]["family"],
                **_custom_model_params(result),
                "model_params": result["model"]["params"],
                "seed": result.get("seed"),
                "effective_seed": result.get("effective_seed"),
                "primary_metric": config.evaluation.primary_metric,
            }
            metrics = dict(result.get("metrics", {}))
            if "training_seconds" in result:
                metrics["training_seconds"] = result["training_seconds"]
            cv_std, cv_folds, cv_params = _cv_metrics(result)
            metrics.update(cv_std)
            params.update(cv_params)
            tags = {
                "phase": "train",
                "candidate_id": result["candidate_id"],
                "status": result["status"],
            }
            if failed:
                tags["error_message"] = result.get("error_message", "")
            run_ids.append(
                tracker.log_run(
                    run_name=result["candidate_id"],
                    params=params,
                    metrics=metrics,
                    tags=tags,
                    failed=failed,
                    fold_metrics=cv_folds,
                )
            )
        return run_ids

    return _safely(config, log)


def log_tuning(config: MLToolConfig, result_set: Any):
    """One run per tuned Phase-5 candidate."""
    hpo = config.hpo

    def log(tracker: _Tracker) -> list[str]:
        run_ids = []
        for result in result_set.candidate_results:
            failed = result["status"] != "SUCCEEDED"
            params: dict[str, Any] = {
                "feature_set": result["feature_set"],
                "model_name": result["model"]["name"],
                "model_family": result["model"]["family"],
                **_custom_model_params(result),
                "num_trials": hpo.num_trials if hpo else None,
                "top_n": hpo.top_n if hpo else None,
                "hpo_time_limit_seconds": hpo.time_limit_seconds if hpo else None,
                "seed": result.get("seed"),
                "effective_seed": result.get("effective_seed"),
                "primary_metric": config.evaluation.primary_metric,
                "searcher": hpo.searcher if hpo else None,
                "searcher_seed": (result.get("hpo") or {}).get("searcher_seed"),
                **_flatten("hp", result.get("best_hyperparameters")),
                **_search_space_params(result),
            }
            metrics = dict(result.get("metrics", {}))
            if "phase4_primary_score" in result:
                metrics["phase4_primary_score"] = result["phase4_primary_score"]
            cv_std, cv_folds, cv_params = _cv_metrics(result)
            metrics.update(cv_std)
            params.update(cv_params)
            for name, value in (result.get("holdout_metrics") or {}).items():
                metrics[f"holdout_{name}"] = value
            tags = {
                "phase": "tune",
                "candidate_id": result["candidate_id"],
                "status": result["status"],
                "hpo_effective": str(result.get("hpo_effective")).lower(),
                "carried_over": str(bool(result.get("carried_over_from_training"))).lower(),
            }
            if failed:
                tags["error_message"] = result.get("error_message", "")
            run_ids.append(
                tracker.log_run(
                    run_name=result["candidate_id"],
                    params=params,
                    metrics=metrics,
                    tags=tags,
                    failed=failed,
                    fold_metrics=cv_folds,
                )
            )
        return run_ids

    return _safely(config, log)


def log_final(config: MLToolConfig, final: Any):
    """Exactly one run for a successful finalize."""

    def log(tracker: _Tracker) -> list[str]:
        result = final.result
        params = {
            "candidate_id": result["candidate_id"],
            "feature_set": result["feature_set"],
            "model_name": result["model"]["name"],
            "model_family": result["model"]["family"],
            **_custom_model_params(result),
            "seed": result.get("seed"),
            "effective_seed": result.get("effective_seed"),
            "primary_metric": result["primary_metric"],
            "train_validation_rows": result["rows"]["train_validation"],
            "test_rows": result["rows"]["test"],
            "searcher_seed": result.get("searcher_seed"),
            **_flatten("hp", result.get("best_hyperparameters")),
            **_search_space_params(result),
        }
        metrics = {f"test_{name}": value for name, value in result["metrics"].items()}
        metrics["selected_validation_score"] = result["selected_validation_score"]
        tags = {
            "phase": "final",
            "candidate_id": result["candidate_id"],
            "test_data_used": "true",
            "status": result["status"],
            "hpo_effective": str(result.get("hpo_effective")).lower(),
        }
        if result.get("hpo_warning"):
            tags["hpo_warning"] = result["hpo_warning"]
        return [
            tracker.log_run(
                run_name=f"final__{result['candidate_id']}",
                params=params,
                metrics=metrics,
                tags=tags,
            )
        ]

    return _safely(config, log)
