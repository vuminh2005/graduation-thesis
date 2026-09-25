"""Phase 13: ``mltool run``, the whole pipeline in one command.

Decisions come from ``status``'s freshness; finalize is never re-run on its own
(a stale final model stops the run unless --refinalize). Uses Phase 12's fake
pipeline: a rule "predictor" per fit, counted in ``RuleAdapter.calls``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from mltool.autogluon_adapter import AutoGluonError
from mltool.cli import init_project, run_project
from mltool.config import load_config
from mltool.registry import list_versions
from mltool.reporting import _phase_states
from mltool.runner import STEPS, plan_decisions
from mltool.state import list_runs
from test_phase12 import (  # noqa: F401  (fake_adapters is an autouse fixture)
    RuleAdapter,
    dataset,
    edit,
    fake_adapters,
    raw_config,
    replace_text,
    write,
)

ALL_RUN = {step: "run" for step in STEPS}


def project(root: Path) -> Path:
    return write(root, raw_config())


def dry(path: Path, **kw: Any) -> dict[str, str]:
    return {d.step: d.action for d in plan_decisions(load_config(path), until=kw.get("until"),
                                                    refinalize=kw.get("refinalize", False))}


def last_run_row(root: Path) -> Any:
    return next(row for row in list_runs(root) if row.command == "run")


def taken(root: Path) -> dict[str, str]:
    """The decisions the last real `mltool run` recorded in SQLite."""
    return {s["step"]: s["action"] for s in last_run_row(root).details["steps"]}


def fresh(path: Path) -> set[str]:
    return {state for phase, (state, _) in _phase_states(load_config(path)).items() if phase not in {"validate", "plan"}}


def finished(root: Path) -> Path:
    path = project(root)
    assert run_project(path) == 0
    RuleAdapter.calls = []
    return path


def test_a_fresh_project_runs_every_step_in_order(tmp_path: Path) -> None:
    path = project(tmp_path)
    assert dry(path) == ALL_RUN
    assert run_project(path) == 0
    assert taken(tmp_path) == ALL_RUN
    step_rows = [row.command for row in reversed(list_runs(tmp_path)) if row.command != "run"]
    # each step logs its own row, in order; a first validate creates no .mltool/
    # and so, as for the command on its own, is not recorded
    assert step_rows == STEPS[1:]
    row = last_run_row(tmp_path)
    assert row.status == "SUCCEEDED" and row.details["failed_step"] is None
    assert RuleAdapter.calls and list_versions(tmp_path) == [1]
    assert fresh(path) == {"fresh"}


def test_a_second_run_with_no_change_skips_everything_and_fits_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = finished(tmp_path)
    expected = {**{s: "skip" for s in STEPS}, "validate": "run", "plan": "run"}
    assert dry(path) == expected
    capsys.readouterr()
    assert run_project(path) == 0
    assert taken(tmp_path) == expected
    assert RuleAdapter.calls == [] and list_versions(tmp_path) == [1]
    out = capsys.readouterr().out
    for text in ("MLTool run summary", "Selected: eng__gbm", "Test metrics: ", "roc_auc=",
                 "Registry version: 1", "mltool score", "mlflow ui --backend-store-uri .mltool/mlflow"):
        assert text in out, text


def _param_change(path: Path) -> None:
    edit(path, lambda raw: raw["models"][1]["params"].__setitem__("max_depth", 5))


def test_a_model_change_reruns_train_and_tune_then_stops_before_finalize(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = finished(tmp_path)
    _param_change(path)
    expected = {"validate": "run", "prepare": "skip", "features": "skip", "plan": "run",
                "train": "run", "tune": "run", "finalize": "stop"}
    assert dry(path) == expected
    final_before = (tmp_path / ".mltool/final/manifest.json").read_bytes()
    capsys.readouterr()
    assert run_project(path) == 2
    assert taken(tmp_path) == expected
    out = capsys.readouterr().out
    assert "evaluates the test split" in out and "--refinalize" in out
    assert "Stopped before finalize" in out and "Result: STOPPED" in out
    assert all("test_features" not in call for call in RuleAdapter.calls)  # no final fit
    assert (tmp_path / ".mltool/final/manifest.json").read_bytes() == final_before
    assert list_versions(tmp_path) == [1]
    assert last_run_row(tmp_path).status == "BLOCKED"

    RuleAdapter.calls = []
    assert dry(path, refinalize=True) == {**{s: "skip" for s in STEPS}, "validate": "run",
                                          "plan": "run", "finalize": "run", "register": "run"}
    assert run_project(path, refinalize=True) == 0
    assert taken(tmp_path)["finalize"] == "run" and taken(tmp_path)["register"] == "run"
    assert len(RuleAdapter.calls) == 1 and "test_features" in RuleAdapter.calls[0]  # one refit
    assert list_versions(tmp_path) == [1, 2] and fresh(path) == {"fresh"}
    finalize_row = next(row for row in list_runs(tmp_path) if row.command == "finalize")
    register_row = next(row for row in list_runs(tmp_path) if row.command == "register")
    assert finalize_row.details["forced"] is True and register_row.details["forced"] is False


def test_refinalize_leaves_a_fresh_final_model_alone(tmp_path: Path) -> None:
    path = finished(tmp_path)
    assert run_project(path, refinalize=True) == 0
    assert taken(tmp_path)["finalize"] == "skip" and RuleAdapter.calls == []


@pytest.mark.parametrize(
    ("edit_source", "first_rerun"),
    [
        (lambda root: replace_text(root / "plugins.py", "- self.mean_", "- self.mean_ + 1.0"), "features"),
        (lambda root: replace_text(root / "pre.py", "* 2.0", "* 3.0"), "prepare"),
    ],
    ids=["plugin_source", "preprocessor_source"],
)
def test_a_source_edit_reruns_from_its_step(
    tmp_path: Path, edit_source: Callable[[Path], None], first_rerun: str
) -> None:
    path = finished(tmp_path)
    edit_source(tmp_path)
    upstream = STEPS[: STEPS.index(first_rerun)]
    expected = {step: ("run" if step in {"validate", "plan"} else "skip") for step in upstream}
    expected.update({step: "run" for step in STEPS[STEPS.index(first_rerun):STEPS.index("finalize")]})
    expected["finalize"] = "stop"
    assert dry(path) == expected
    assert run_project(path) == 2
    assert taken(tmp_path) == expected
    assert run_project(path, refinalize=True) == 0 and fresh(path) == {"fresh"}


def test_until_stops_after_the_named_step(tmp_path: Path) -> None:
    path = project(tmp_path)
    assert dry(path, until="tune") == {step: "run" for step in STEPS[: STEPS.index("tune") + 1]}
    assert run_project(path, until="tune") == 0
    assert list(taken(tmp_path)) == STEPS[: STEPS.index("tune") + 1]
    assert not (tmp_path / ".mltool/final").exists()
    states = _phase_states(load_config(path))
    assert states["tune"][0] == "fresh" and states["finalize"][0] == "missing"
    RuleAdapter.calls = []
    assert run_project(path) == 0  # continues from where it stopped
    assert taken(tmp_path)["train"] == "skip" and taken(tmp_path)["finalize"] == "run"


def test_a_dry_run_fits_nothing_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = project(tmp_path)
    capsys.readouterr()
    assert run_project(path, dry_run=True) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "prepare   run   missing" in out
    assert not (tmp_path / ".mltool").exists() and RuleAdapter.calls == []


def test_a_failing_middle_step_stops_the_run_with_its_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = project(tmp_path)

    class Failing:
        def fit_predict(self, **kw: Any) -> Any:
            raise AutoGluonError("every candidate fails")

    from mltool.training import train_experiment

    monkeypatch.setattr("mltool.cli.train_experiment",
                        lambda plan: train_experiment(plan, adapter_factory=Failing))
    capsys.readouterr()
    assert run_project(path) == 1  # train's own exit code when every candidate fails
    assert list(taken(tmp_path)) == STEPS[: STEPS.index("train") + 1]
    row = last_run_row(tmp_path)
    assert row.status == "FAILED" and row.details["failed_step"] == "train"
    assert "Result: FAILED at train (exit 1)" in capsys.readouterr().out
    assert not (tmp_path / ".mltool/tuning").exists()


def test_disabled_validation_is_skipped_not_failed(tmp_path: Path) -> None:
    path = project(tmp_path)
    edit(path, lambda raw: raw["validation"].__setitem__("enabled", False))
    assert dry(path)["validate"] == "skip"
    assert run_project(path) == 0 and taken(tmp_path)["validate"] == "skip"


SCENARIOS: dict[str, Callable[[Path], None]] = {
    "none": lambda path: None,
    "model_param": _param_change,
    "plugin_source": lambda path: replace_text(path.parent / "plugins.py", "- self.mean_", "- self.mean_ + 1.0"),
    "preprocessor_source": lambda path: replace_text(path.parent / "pre.py", "* 2.0", "* 3.0"),
    "split_seed": lambda path: edit(path, lambda raw: raw["split"].__setitem__("random_seed", 43)),
}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_run_decisions_agree_with_status(tmp_path: Path, scenario: str) -> None:
    path = finished(tmp_path)
    SCENARIOS[scenario](path)
    states = _phase_states(load_config(path))
    decisions = dry(path)
    for step, action in decisions.items():
        if step in {"validate", "plan"}:
            continue
        state = states[step][0]
        if step == "finalize":
            assert action == {"fresh": "skip", "missing": "run", "stale": "stop"}[state]
        else:
            assert action == ("skip" if state == "fresh" else "run"), step
    run_project(path)
    assert taken(tmp_path) == decisions  # the real run took the decisions the dry run showed


def test_a_project_fresh_from_init_runs_end_to_end(tmp_path: Path) -> None:
    """init -> put the data in place -> run, with no other edit."""
    assert init_project(tmp_path) == 0
    dataset().to_csv(tmp_path / "data/dataset.csv", index=False)  # target "label", like the template
    path = tmp_path / "mltool.yaml"
    assert run_project(path) == 0
    assert taken(tmp_path) == ALL_RUN and list_versions(tmp_path) == [1]
