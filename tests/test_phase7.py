from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Any

import pytest
import yaml

import mltool.tracking as tracking
from mltool.cli import (
    best_project,
    final_result_project,
    finalize_project,
    leaderboard_project,
    logs_project,
    plan_project,
    register_project,
    status_project,
    train_project,
    tune_project,
    validate_project,
)
from mltool.config import load_config
from mltool.experiment import build_experiment_plan
from mltool.finalize import finalize_experiment
from mltool.registry import list_versions
from mltool.state import last_run_per_command, list_runs, state_path
from mltool.training import train_experiment
from mltool.tuning import tune_experiment
from test_phase4 import RecordingAdapter, materialize, project_config
from test_phase5 import TuningAdapter, hpo_raw
from test_phase6 import MODELS, FinalAdapter

_train, _tune, _finalize = train_experiment, tune_experiment, finalize_experiment


@pytest.fixture(autouse=True)
def fake_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    RecordingAdapter.failures = set()
    TuningAdapter.failures = set()
    monkeypatch.setattr(
        "mltool.cli.train_experiment", lambda plan: _train(plan, adapter_factory=RecordingAdapter)
    )
    monkeypatch.setattr(
        "mltool.cli.tune_experiment",
        lambda plan, sel: _tune(plan, sel, adapter_factory=TuningAdapter),
    )
    monkeypatch.setattr(
        "mltool.cli.finalize_experiment",
        lambda plan, inp: _finalize(plan, inp, adapter_factory=FinalAdapter),
    )
    FinalAdapter.calls = []


def project(root: Path, *, through: str = "finalize") -> Path:
    """Run the CLI-level pipeline (prepare/features already logged by ``materialize``)."""
    path = materialize(root, raw=project_config(models=MODELS, training={"seed": 7}))
    hpo_raw(path, top_n=2)
    steps = ["plan", "train", "tune", "finalize"]
    fns = {"plan": plan_project, "train": train_project, "tune": tune_project,
           "finalize": finalize_project}
    for step in steps[: steps.index(through) + 1]:
        assert fns[step](path) == 0, step
    return path


def mlflow_runs(root: Path) -> list[Any]:
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=tracking.tracking_uri(root))
    experiment = client.get_experiment_by_name("phase4-test")
    if experiment is None:
        return []
    return list(client.search_runs([experiment.experiment_id], max_results=1000))


def by_phase(root: Path, phase: str) -> list[Any]:
    return [run for run in mlflow_runs(root) if run.data.tags.get("phase") == phase]


# --- SQLite state ---------------------------------------------------------------------


def test_every_state_changing_command_writes_a_row(tmp_path: Path) -> None:
    project(tmp_path)
    assert validate_project(tmp_path / "mltool.yaml") == 0
    assert register_project(tmp_path / "mltool.yaml") == 0
    runs = list(reversed(list_runs(tmp_path)))
    assert [r.command for r in runs] == [
        "prepare", "features", "plan", "train", "tune", "finalize", "validate", "register",
    ]
    assert all(r.status == "SUCCEEDED" and r.exit_code == 0 for r in runs)
    assert all(r.finished_at and r.started_at <= r.finished_at for r in runs)
    details = {r.command: r.details for r in runs}
    assert details["train"]["candidates"] == 2 and details["train"]["succeeded"] == 2
    assert details["train"]["primary_metric"] == "roc_auc" and details["train"]["mlflow_runs"] == 2
    assert details["tune"]["candidates"] == 2 and details["tune"]["mlflow_runs"] == 2
    assert details["tune"]["selected_candidate_id"]
    assert details["plan"]["candidates"] == 2
    assert details["finalize"]["forced"] is False and details["finalize"]["mlflow_runs"] == 1
    assert details["finalize"]["test_metrics"]["roc_auc"] == 1.0
    assert details["prepare"]["test_rows"] == 15
    assert details["register"]["version"] == 1


def test_schema_and_failed_commands_are_logged(tmp_path: Path) -> None:
    path = materialize(tmp_path, raw=project_config(models=MODELS))
    hpo_raw(path)
    assert tune_project(path) == 2  # no training artifacts yet
    (row,) = [r for r in list_runs(tmp_path) if r.command == "tune"]
    assert row.status == "FAILED" and row.exit_code == 2
    assert 'run "mltool train" first' in row.details["error"]
    connection = sqlite3.connect(state_path(tmp_path))
    columns = [c[1] for c in connection.execute("PRAGMA table_info(runs)")]
    connection.close()
    assert columns == ["id", "command", "started_at", "finished_at", "status", "exit_code", "details"]


def test_unexpected_exception_is_logged_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = materialize(tmp_path, raw=project_config(models=MODELS))
    monkeypatch.setattr("mltool.cli.train_experiment", lambda plan: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        train_project(path)
    row = list_runs(tmp_path)[0]
    assert row.command == "train" and row.status == "FAILED" and "division" in row.details["error"]


def test_state_is_not_created_when_nothing_was_created(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    path = tmp_path / "mltool.yaml"
    path.write_text(yaml.safe_dump(project_config(models=MODELS)))
    assert validate_project(path) == 2  # dataset missing
    assert not (tmp_path / ".mltool").exists()


# --- finalize block / force ---------------------------------------------------------------


def test_second_finalize_is_blocked_logged_and_creates_no_mlflow_run(tmp_path: Path) -> None:
    path = project(tmp_path)
    result_before = (tmp_path / ".mltool/final/result.json").read_bytes()
    runs_before = len(mlflow_runs(tmp_path))
    fits_before = len(FinalAdapter.calls)

    assert finalize_project(path) == 2
    assert (tmp_path / ".mltool/final/result.json").read_bytes() == result_before
    assert len(FinalAdapter.calls) == fits_before  # nothing was refit or evaluated
    assert len(mlflow_runs(tmp_path)) == runs_before
    row = list_runs(tmp_path)[0]
    assert (row.command, row.status, row.exit_code) == ("finalize", "BLOCKED", 2)
    assert row.details["blocked"] is True and row.details["forced"] is False
    assert "--force" in row.details["error"]


def test_blocked_message_is_printed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = project(tmp_path)
    capsys.readouterr()
    finalize_project(path)
    out = capsys.readouterr().out
    assert "a final model already exists" in out and "--force" in out and "register" in out


def test_forced_finalize_runs_again_and_is_logged(tmp_path: Path) -> None:
    path = project(tmp_path)
    assert finalize_project(path, force=True) == 0
    assert len(FinalAdapter.calls) == 2
    row = list_runs(tmp_path)[0]
    assert (row.command, row.status) == ("finalize", "SUCCEEDED")
    assert row.details["forced"] is True and row.details["mlflow_runs"] == 1
    assert len(by_phase(tmp_path, "final")) == 2  # first run + forced run
    statuses = [r.status for r in list_runs(tmp_path) if r.command == "finalize"]
    assert statuses == ["SUCCEEDED", "SUCCEEDED"]


def test_first_finalize_needs_no_force_and_force_flag_is_recorded_false(tmp_path: Path) -> None:
    project(tmp_path)
    (row,) = [r for r in list_runs(tmp_path) if r.command == "finalize"]
    assert row.details["forced"] is False


# --- MLflow ---------------------------------------------------------------------------------


def test_train_creates_one_mlflow_run_per_candidate(tmp_path: Path) -> None:
    project(tmp_path, through="train")
    runs = by_phase(tmp_path, "train")
    assert {r.data.tags["candidate_id"] for r in runs} == {"base__lightgbm", "base__forest"}
    run = next(r for r in runs if r.data.tags["candidate_id"] == "base__lightgbm")
    assert run.data.tags["status"] == "SUCCEEDED" and run.info.status == "FINISHED"
    assert run.data.params["model_family"] == "GBM"
    assert run.data.params["seed"] == "7" and run.data.params["effective_seed"] == "7"
    assert run.data.params["feature_set"] == "base"
    assert run.data.params["primary_metric"] == "roc_auc"
    assert {"roc_auc", "f1", "accuracy"} <= set(run.data.metrics)
    assert run.data.metrics["roc_auc"] == 1.0
    assert run.data.params["model_params"] == "{}"


def test_failed_candidate_is_tracked_as_a_failed_mlflow_run(tmp_path: Path) -> None:
    path = materialize(tmp_path, raw=project_config(models=MODELS, training={"seed": 7}))
    RecordingAdapter.failures = {"forest"}
    assert train_project(path) == 0
    failed = [r for r in by_phase(tmp_path, "train") if r.data.tags["status"] == "FAILED"]
    assert len(failed) == 1 and failed[0].info.status == "FAILED"
    assert "intentional failure" in failed[0].data.tags["error_message"]
    assert not failed[0].data.metrics


def test_tune_creates_one_run_per_tuned_candidate(tmp_path: Path) -> None:
    project(tmp_path, through="tune")
    runs = by_phase(tmp_path, "tune")
    assert len(runs) == 2
    run = next(r for r in runs if r.data.params["model_family"] == "GBM")  # RF is carried over
    assert run.data.params["num_trials"] == "4" and run.data.params["top_n"] == "2"
    assert run.data.params["hpo_time_limit_seconds"] == "30"
    assert run.data.params["hp.learning_rate"] == "0.05"  # best_hyperparameters, flattened
    assert "hp.seed" in run.data.params and "effective_seed" in run.data.params
    assert run.data.tags["hpo_effective"] in {"true", "false"}
    assert run.data.metrics["roc_auc"] == 1.0 and "phase4_primary_score" in run.data.metrics
    assert {r.data.tags["candidate_id"] for r in runs} == {
        json.loads(p.read_text())["candidate_id"]
        for p in (tmp_path / ".mltool/tuning/candidates").glob("*/result.json")
    }


def test_successful_finalize_logs_exactly_one_run(tmp_path: Path) -> None:
    project(tmp_path)
    (run,) = by_phase(tmp_path, "final")
    selected = json.loads((tmp_path / ".mltool/tuning/selected.json").read_text())
    assert run.data.tags["test_data_used"] == "true"
    assert run.data.params["candidate_id"] == selected["candidate_id"]
    assert run.data.params["effective_seed"] == "7"
    assert run.data.params["hp.learning_rate"] == "0.05"
    assert run.data.metrics["test_roc_auc"] == 1.0 and "selected_validation_score" in run.data.metrics
    # train/tune runs never claim test data
    assert all(r.data.tags.get("test_data_used") is None
               for r in mlflow_runs(tmp_path) if r.data.tags["phase"] != "final")


def test_mlflow_uses_a_project_local_file_store(tmp_path: Path) -> None:
    project(tmp_path, through="train")
    store = tmp_path / ".mltool/mlflow"
    assert store.is_dir() and any(store.iterdir())
    assert tracking.tracking_uri(tmp_path).startswith("file:///")
    assert tracking.tracking_uri(tmp_path).endswith("/.mltool/mlflow")


def test_mlflow_failure_never_aborts_a_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = materialize(tmp_path, raw=project_config(models=MODELS))

    def broken(self: Any, config: Any) -> None:
        raise RuntimeError("tracking store unavailable")

    monkeypatch.setattr(tracking._Tracker, "__init__", broken)
    assert train_project(path) == 0
    captured = capsys.readouterr()
    assert "MLflow tracking failed and was skipped: tracking store unavailable" in captured.err
    assert (tmp_path / ".mltool/training/leaderboard.json").is_file()
    row = list_runs(tmp_path)[0]
    assert row.status == "SUCCEEDED" and row.details["mlflow_runs"] == 0


# --- registry ---------------------------------------------------------------------------------


def test_register_copies_artifacts_and_writes_metadata(tmp_path: Path) -> None:
    from test_phase6 import final_project  # plugins + preprocessor => every artifact exists

    path, _ = final_project(tmp_path, plugins=True)
    plan = build_experiment_plan(load_config(path))
    from mltool.finalize import load_finalize_input

    _finalize(plan, load_finalize_input(plan), adapter_factory=FinalAdapter)
    assert register_project(path) == 0
    entry = tmp_path / ".mltool/registry/1"
    final = tmp_path / ".mltool/final"
    for name in ("predictor/fake.txt", "preprocessor.pkl", "result.json", "manifest.json"):
        assert (entry / name).read_bytes() == (final / name).read_bytes(), name
    (plugin_pkl,) = list((entry / "feature_plugins").glob("*.pkl"))
    assert plugin_pkl.read_bytes() == (final / "feature_plugins" / plugin_pkl.name).read_bytes()
    meta = json.loads((entry / "metadata.json").read_text())
    result = json.loads((final / "result.json").read_text())
    assert meta["version"] == 1 and meta["registered_at"]
    assert meta["selected"]["candidate_id"] == result["candidate_id"]
    assert meta["best_hyperparameters"] == result["best_hyperparameters"]
    assert meta["effective_seed"] == result["effective_seed"] == 7
    assert meta["test_metrics"] == result["metrics"]
    assert len(meta["config_fingerprint"]) == 64 and meta["git_commit"] is None
    assert "predictor" in meta["artifacts"] and "metadata.json" in meta["artifacts"]
    row = list_runs(tmp_path)[0]
    assert (row.command, row.status, row.details["version"]) == ("register", "SUCCEEDED", 1)


def test_register_refuses_without_final_and_logs_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = project(tmp_path, through="tune")
    capsys.readouterr()
    assert register_project(path) == 2
    assert 'run "mltool finalize" first' in capsys.readouterr().out
    assert not (tmp_path / ".mltool/registry").exists() or not list_versions(tmp_path)
    row = list_runs(tmp_path)[0]
    assert (row.command, row.status) == ("register", "FAILED")


def test_multiple_registrations_keep_history_across_forced_finalize(tmp_path: Path) -> None:
    path = project(tmp_path)
    assert register_project(path) == 0
    assert register_project(path) == 0
    assert finalize_project(path, force=True) == 0
    assert register_project(path) == 0
    assert list_versions(tmp_path) == [1, 2, 3]
    # versions stay monotonic even if an old one is removed
    import shutil

    shutil.rmtree(tmp_path / ".mltool/registry/2")
    assert register_project(path) == 0
    assert list_versions(tmp_path) == [1, 3, 4]
    assert not list((tmp_path / ".mltool/registry").glob(".registry-staging-*"))


def test_register_records_git_commit_when_available(tmp_path: Path) -> None:
    path = project(tmp_path)
    git = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("-c", "user.email=a@b.c", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x")
    expected = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True,
                              text=True).stdout.strip()
    assert register_project(path) == 0
    meta = json.loads((tmp_path / ".mltool/registry/1/metadata.json").read_text())
    assert meta["git_commit"] == expected


# --- status / logs / best -----------------------------------------------------------------------


def test_status_reports_freshness_and_last_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = project(tmp_path)
    capsys.readouterr()
    assert status_project(path) == 0
    out = capsys.readouterr().out
    rows = {line.split()[0]: line.split() for line in out.splitlines() if line.split()[:1]}
    for phase in ("prepare", "features", "train", "tune", "finalize"):
        assert rows[phase][1] == "fresh", phase
        assert rows[phase][2] == "SUCCEEDED"
    assert rows["register"][1] == "missing" and rows["register"][2] == "never"
    assert rows["plan"][1] == "n/a" and rows["plan"][2] == "SUCCEEDED"

    raw = yaml.safe_load(path.read_text())
    raw["models"][0]["params"] = {"num_boost_round": 3}
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    capsys.readouterr()
    assert status_project(path) == 0
    out = capsys.readouterr().out
    rows = {line.split()[0]: line.split() for line in out.splitlines() if line.split()[:1]}
    assert rows["prepare"][1] == "fresh"
    assert rows["train"][1] == "stale" and rows["tune"][1] == "stale"
    assert "Notes" in out and "train:" in out


def test_status_on_an_empty_project_shows_missing_and_creates_nothing(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    path = tmp_path / "mltool.yaml"
    path.write_text(yaml.safe_dump(project_config(models=MODELS)))
    assert status_project(path) == 0
    assert not (tmp_path / ".mltool").exists()


def test_logs_are_most_recent_first_and_limited(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = project(tmp_path)
    finalize_project(path)  # BLOCKED
    capsys.readouterr()
    assert logs_project(path) == 0
    lines = capsys.readouterr().out.splitlines()
    body = [line.split()[:3] for line in lines[3:]]
    assert body[0][1:] == ["finalize", "BLOCKED"] and body[1][1:] == ["finalize", "SUCCEEDED"]
    assert [b[1] for b in body] == [
        "finalize", "finalize", "tune", "train", "plan", "features", "prepare"
    ]
    ids = [int(b[0]) for b in body]
    assert ids == sorted(ids, reverse=True)
    capsys.readouterr()
    assert logs_project(path, limit=2) == 0
    limited = capsys.readouterr().out.splitlines()[3:]
    assert len(limited) == 2


def test_logs_with_no_history(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "data").mkdir()
    path = tmp_path / "mltool.yaml"
    path.write_text(yaml.safe_dump(project_config(models=MODELS)))
    assert logs_project(path) == 0
    assert "No runs have been recorded yet." in capsys.readouterr().out


def test_best_requires_a_final_or_registered_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = project(tmp_path, through="tune")
    capsys.readouterr()
    assert best_project(path) == 2
    assert 'run "mltool finalize" first' in capsys.readouterr().out


def test_best_shows_final_and_latest_registered_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = project(tmp_path)
    capsys.readouterr()
    assert best_project(path) == 0
    out = capsys.readouterr().out
    assert "Finalized model" in out and "roc_auc=1.000000" in out
    assert "Latest registered version" not in out
    register_project(path)
    register_project(path)
    capsys.readouterr()
    assert best_project(path) == 0
    out = capsys.readouterr().out
    assert "Latest registered version: 2" in out and "registered versions: 2" in out


def test_reporting_commands_are_read_only(tmp_path: Path) -> None:
    path = project(tmp_path)
    register_project(path)
    db = state_path(tmp_path)
    before = (db.read_bytes(), len(list_runs(tmp_path)))
    snapshot = {p: p.read_bytes() for p in (tmp_path / ".mltool/final").glob("*.json")}
    for command in (status_project, logs_project, best_project, final_result_project,
                    leaderboard_project):
        assert command(path) == 0
    assert (db.read_bytes(), len(list_runs(tmp_path))) == before
    assert {p: p.read_bytes() for p in snapshot} == snapshot
    assert list_versions(tmp_path) == [1]
    assert set(last_run_per_command(tmp_path)) == {
        "prepare", "features", "plan", "train", "tune", "finalize", "register"
    }  # read-only commands never log
