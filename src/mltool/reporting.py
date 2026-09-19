"""Phase-7 read-only reporting: ``status``, ``logs`` and ``best``. Nothing here writes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mltool.config import MLToolConfig
from mltool.experiment import ExperimentError, build_experiment_plan, load_feature_artifacts
from mltool.finalize import (
    FinalizeError,
    _validate_prepared_manifest,
    load_finalize_input,
    load_persisted_final,
)
from mltool.registry import RegistryError, list_versions, load_metadata
from mltool.state import RunRecord, last_run_per_command, list_runs
from mltool.training import TrainingError
from mltool.tuning import TuningError, _read_training_artifacts, _validate_training_freshness

PHASES = ["validate", "prepare", "features", "plan", "train", "tune", "finalize", "register"]


def _phase_states(config: MLToolConfig) -> dict[str, tuple[str, str]]:
    """Map phase -> (state, note); state is one of missing/fresh/stale/n/a."""
    mltool = config.config_path.parent / ".mltool"
    states: dict[str, tuple[str, str]] = {
        "validate": ("n/a", "no artifacts"),
        "plan": ("n/a", "no artifacts"),
    }

    if not (mltool / "prepared/manifest.json").is_file():
        states["prepare"] = ("missing", "")
    else:
        try:
            _validate_prepared_manifest(config)
            states["prepare"] = ("fresh", "")
        except FinalizeError as exc:
            states["prepare"] = ("stale", str(exc))

    plan = None
    if not (mltool / "features/manifest.json").is_file():
        states["features"] = ("missing", "")
    else:
        try:
            load_feature_artifacts(config)
            states["features"] = ("fresh", "")
            plan = build_experiment_plan(config)
        except ExperimentError as exc:
            states["features"] = ("stale", str(exc))

    if not (mltool / "training/manifest.json").is_file():
        states["train"] = ("missing", "")
    elif plan is None:
        states["train"] = ("stale", "feature artifacts are missing or stale")
    else:
        try:
            manifest, _ = _read_training_artifacts(config)
            _validate_training_freshness(plan, manifest)
            states["train"] = ("fresh", "")
        except (TrainingError, TuningError) as exc:
            states["train"] = ("stale", str(exc))

    if not (mltool / "tuning/manifest.json").is_file():
        states["tune"] = ("missing", "")
    elif plan is None:
        states["tune"] = ("stale", "feature artifacts are missing or stale")
    else:
        try:
            load_finalize_input(plan)
            states["tune"] = ("fresh", "")
        except (TrainingError, TuningError, FinalizeError) as exc:
            states["tune"] = ("stale", str(exc))

    if not (mltool / "final/manifest.json").is_file():
        states["finalize"] = ("missing", "")
    else:
        try:
            warning = load_persisted_final(config).warning
            states["finalize"] = ("stale", warning) if warning else ("fresh", "")
        except FinalizeError as exc:
            states["finalize"] = ("stale", str(exc))

    versions = list_versions(config.config_path.parent)
    states["register"] = (
        ("fresh", f"{len(versions)} version(s), latest {versions[-1]}") if versions else ("missing", "")
    )
    return states


def render_status(config: MLToolConfig) -> str:
    states = _phase_states(config)
    last = last_run_per_command(config.config_path.parent)
    lines = ["MLTool status", "", f"{'Phase':<10} {'Artifacts':<10} {'Last run':<32}", ""]
    notes: list[str] = []
    for phase in PHASES:
        state, note = states.get(phase, ("n/a", ""))
        run = last.get(phase)
        last_text = f"{run.status} {run.started_at}" if run else "never"
        lines.append(f"{phase:<10} {state:<10} {last_text}")
        if note and state in {"stale", "fresh"}:
            notes.append(f"  {phase}: {note}")
    if notes:
        lines.extend(["", "Notes", *notes])
    return "\n".join(lines)


def _summary(details: dict[str, Any], limit: int = 70) -> str:
    keys = [
        "blocked", "forced", "candidate_id", "selected_candidate_id", "primary_metric",
        "candidates", "succeeded", "failed", "version", "mlflow_runs", "error",
    ]
    parts = [f"{key}={details[key]}" for key in keys if key in details]
    text = " ".join(parts)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def render_logs(project_root: Path, limit: int | None = 20) -> str:
    runs: list[RunRecord] = list_runs(project_root, limit)
    lines = ["MLTool run history", ""]
    if not runs:
        return "\n".join([*lines, "No runs have been recorded yet."])
    lines.append(f"{'ID':<4} {'Command':<9} {'Status':<10} {'Started (UTC)':<26} Details")
    for run in runs:
        lines.append(
            f"{run.id:<4} {run.command:<9} {run.status:<10} {run.started_at:<26} "
            f"{_summary(run.details)}"
        )
    return "\n".join(lines)


def render_best(config: MLToolConfig) -> str:
    root = config.config_path.parent
    final = None
    if (root / ".mltool/final/manifest.json").is_file():
        final = load_persisted_final(config)
    versions = list_versions(root)
    if final is None and not versions:
        raise FinalizeError('no final or registered model was found; run "mltool finalize" first')
    lines = ["MLTool best configuration", ""]
    if final is not None:
        lines.extend(["Finalized model (.mltool/final)", final.render().split("\n", 2)[2].rstrip()])
    if versions:
        try:
            meta = load_metadata(root, versions[-1])
        except RegistryError as exc:
            raise FinalizeError(str(exc)) from exc
        metrics = ", ".join(f"{k}={v:.6f}" for k, v in meta["test_metrics"].items())
        lines.extend(
            [
                "",
                f"Latest registered version: {meta['version']} ({meta['registered_at']})",
                f"  selected: {meta['selected']['candidate_id']}",
                f"  test metrics: {metrics}",
                f"  git commit: {meta['git_commit'] or 'n/a'}  registered versions: {len(versions)}",
            ]
        )
    return "\n".join(lines)
