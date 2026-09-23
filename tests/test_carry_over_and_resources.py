"""Regressions for two findings of the PS4E1 Kaggle experiment.

Fix 1: tune carries over candidates it cannot tune (RF/XT, ENSEMBLE) from
training instead of re-fitting them to reproduce the same scores.
Fix 2: cgroup v2 memory/CPU limits reach AutoGluon, which otherwise sizes its
memory guard from the host total.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import mltool.resources as resources
from mltool.autogluon_adapter import AutoGluonAdapter, AutoGluonError
from mltool.cli import tune_project
from mltool.config import ConfigError, ModelConfig, TaskConfig, TrainingConfig, load_config
from mltool.experiment import build_experiment_plan
from mltool.finalize import finalize_experiment, load_finalize_input
from mltool.resources import (
    GIB,
    CgroupLimits,
    detect_cgroup_limits,  # the real one; conftest patches the module attribute
    resolve_resource_limits,
)
from mltool.training import train_experiment
from mltool.tuning import TuningError, build_tuning_selection, tune_experiment
from test_audit_fixes import build, fake_adapters  # noqa: F401  (autouse fixture)
from test_phase4 import FakePredictor, materialize, project_config, write_project
from test_phase5 import TuningAdapter, hpo_raw, trained_project
from test_phase6 import FinalAdapter
from test_phase7 import by_phase
from test_phase9 import CvFakeAdapter, cv_project, run_train

ENSEMBLE = {"name": "ens", "family": "ENSEMBLE", "params": {}}
GBM = {"name": "gbm", "family": "GBM", "params": {}}
RF = {"name": "rf", "family": "RF", "params": {}}


def training_result(root: Path, cid: str) -> dict[str, Any]:
    return json.loads((root / ".mltool/training/candidates" / cid / "result.json").read_text())


def tuning_result(root: Path, cid: str) -> dict[str, Any]:
    return json.loads((root / ".mltool/tuning/candidates" / cid / "result.json").read_text())


def cv_tune(path: Path) -> Any:
    plan = build_experiment_plan(load_config(path))
    selection = build_tuning_selection(plan)
    CvFakeAdapter.calls = []
    return plan, selection, tune_experiment(plan, selection, adapter_factory=CvFakeAdapter)


# =========================== Fix 1: carry-over ==================================


def test_cv_untunable_candidates_make_no_fit_and_copy_training_exactly(tmp_path: Path) -> None:
    path = cv_project(tmp_path, cv={"folds": 3}, models=[GBM, RF, ENSEMBLE])
    hpo_raw(path, top_n=3)
    run_train(path)
    _, selection, result = cv_tune(path)

    families = {c.candidate_id: c.model.family for c in selection.candidates}
    current_fingerprint = result.cv["fold_fingerprint"]  # this tune's own fold plan
    fitted = {call["model"].name for call in CvFakeAdapter.calls}
    assert fitted == {"gbm"}  # the HPO search plus its CV re-scoring; nothing else
    for cid, family in families.items():
        tuned = tuning_result(tmp_path, cid)
        if family == "GBM":
            assert "carried_over_from_training" not in tuned and tuned["hpo_effective"] is True
            continue
        trained = training_result(tmp_path, cid)
        assert tuned["carried_over_from_training"] is True
        assert tuned["hpo_effective"] is False and tuned["hpo_warning"]
        assert tuned["metrics"] == trained["metrics"]  # exactly, not approximately
        assert tuned["cv"] == trained["cv"]  # per-fold metrics and the fold fingerprint
        assert tuned["cv"]["fold_fingerprint"] == current_fingerprint
        assert tuned["predictor_path"] is None and tuned["predictor_persisted"] is False
        assert tuned["best_hyperparameters"] == trained["model"]["params"]
    manifest = json.loads((tmp_path / ".mltool/tuning/manifest.json").read_text())
    assert set(manifest["carried_over_candidate_ids"]) == {
        cid for cid, family in families.items() if family != "GBM"
    }
    by_id = {row["candidate_id"]: row for row in result.leaderboard}
    for cid in manifest["carried_over_candidate_ids"]:
        assert by_id[cid]["primary_score"] == training_result(tmp_path, cid)["metrics"]["roc_auc"]
    assert "carried over from training (not tunable; no fit)" in result.render()


def test_holdout_untunable_candidate_is_carried_over_too(tmp_path: Path) -> None:
    path = trained_project(tmp_path)  # GBM, RF, GBM; top_n=2 selects one GBM and the RF
    plan = build_experiment_plan(load_config(path))
    selection = build_tuning_selection(plan)
    TuningAdapter.calls, TuningAdapter.failures = [], set()
    tune_experiment(plan, selection, adapter_factory=TuningAdapter)
    rf = next(c for c in selection.candidates if c.model.family == "RF")
    assert all(call["model"].family != "RF" for call in TuningAdapter.calls)
    tuned, trained = tuning_result(tmp_path, rf.candidate_id), training_result(
        tmp_path, rf.candidate_id
    )
    assert tuned["carried_over_from_training"] is True
    assert tuned["metrics"] == trained["metrics"] and "cv" not in tuned


def test_finalize_still_refits_a_carried_over_selection_from_scratch(tmp_path: Path) -> None:
    path = trained_project(tmp_path)
    plan = build_experiment_plan(load_config(path))
    selection = build_tuning_selection(plan)
    # every tunable candidate fails, so the carried-over RF is selected
    TuningAdapter.calls = []
    TuningAdapter.failures = {c.model.name for c in selection.candidates if c.model.family != "RF"}
    tuned = tune_experiment(plan, selection, adapter_factory=TuningAdapter)
    assert tuned.selected["model"]["family"] == "RF"
    assert tuned.selected["hpo_effective"] is False and tuned.selected["best_hyperparameters"] == {}

    FinalAdapter.calls = []
    final = finalize_experiment(plan, load_finalize_input(plan), adapter_factory=FinalAdapter)
    (call,) = FinalAdapter.calls
    assert call["model"].family == "RF" and call["best_hyperparameters"] == {}
    assert final.result["hpo_effective"] is False
    assert "Tuned: no (family RF has no HPO search space)" in final.render()


def test_every_fitted_candidate_failing_still_leaves_no_selection(tmp_path: Path) -> None:
    """The coverage Phase 5 had before RF became carried over: nothing to select."""
    raw = project_config(models=[GBM, {"name": "gbm2", "family": "GBM", "params": {}}])
    path = materialize(tmp_path, raw=raw)
    hpo_raw(path, top_n=2)
    from test_phase4 import RecordingAdapter

    RecordingAdapter.failures, RecordingAdapter.calls = set(), []
    train_experiment(build_experiment_plan(load_config(path)), adapter_factory=RecordingAdapter)
    plan = build_experiment_plan(load_config(path))
    selection = build_tuning_selection(plan)
    TuningAdapter.failures = {c.model.name for c in selection.candidates}
    result = tune_experiment(plan, selection, adapter_factory=TuningAdapter)
    assert not result.is_successful and result.selected is None
    assert not (tmp_path / ".mltool/tuning/selected.json").exists()


def test_a_different_fold_assignment_refuses_carry_over_before_any_fit(tmp_path: Path) -> None:
    path = cv_project(tmp_path, cv={"folds": 3}, models=[GBM, RF])
    hpo_raw(path, top_n=2)
    run_train(path)
    rf_id = next(p.name for p in (tmp_path / ".mltool/training/candidates").iterdir()
                 if p.name.endswith("__rf"))
    result_path = tmp_path / ".mltool/training/candidates" / rf_id / "result.json"
    stored = json.loads(result_path.read_text())
    stored["cv"]["fold_fingerprint"] = "0" * 64
    result_path.write_text(json.dumps(stored), encoding="utf-8")

    plan = build_experiment_plan(load_config(path))
    selection = build_tuning_selection(plan)
    CvFakeAdapter.calls = []
    with pytest.raises(TuningError, match="different fold assignment.*mltool train"):
        tune_experiment(plan, selection, adapter_factory=CvFakeAdapter)
    assert CvFakeAdapter.calls == []  # failed before the tunable GBM spent a fit
    assert not (tmp_path / ".mltool/tuning").exists()


def test_a_changed_seed_refuses_carry_over(tmp_path: Path) -> None:
    path = trained_project(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["training"] = {"seed": 99}
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    plan = build_experiment_plan(load_config(path))
    selection = build_tuning_selection(plan)  # the config signature has no seed
    TuningAdapter.calls = []
    with pytest.raises(TuningError, match="used a different seed"):
        tune_experiment(plan, selection, adapter_factory=TuningAdapter)
    assert TuningAdapter.calls == []


def test_a_missing_training_result_refuses_carry_over(tmp_path: Path) -> None:
    path = trained_project(tmp_path)
    plan = build_experiment_plan(load_config(path))
    selection = build_tuning_selection(plan)
    rf = next(c for c in selection.candidates if c.model.family == "RF")
    (tmp_path / ".mltool/training/candidates" / rf.candidate_id / "result.json").unlink()
    with pytest.raises(TuningError, match="missing or unreadable.*mltool train"):
        tune_experiment(plan, selection, adapter_factory=TuningAdapter)


def test_stale_training_is_still_refused_by_the_existing_check(tmp_path: Path) -> None:
    path = cv_project(tmp_path, cv={"folds": 3}, models=[GBM, RF])
    hpo_raw(path, top_n=2)
    run_train(path)
    raw = yaml.safe_load(path.read_text())
    raw["evaluation"]["cv"] = {"folds": 4}
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(TuningError, match='"cv" differs'):
        build_tuning_selection(build_experiment_plan(load_config(path)))


def test_mlflow_tags_the_carried_over_run_with_the_copied_metrics(tmp_path: Path) -> None:
    build(tmp_path, through="tune")  # GBM + RF, top_n=2
    runs = {r.data.params["model_family"]: r for r in by_phase(tmp_path, "tune")}
    assert runs["RF"].data.tags["carried_over"] == "true"
    assert runs["GBM"].data.tags["carried_over"] == "false"
    rf_id = runs["RF"].data.tags["candidate_id"]
    copied = training_result(tmp_path, rf_id)["metrics"]
    assert {k: runs["RF"].data.metrics[k] for k in copied} == copied


def test_cli_tune_reports_the_carry_over(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = build(tmp_path, through="train")
    capsys.readouterr()
    assert tune_project(path) == 0
    captured = capsys.readouterr()
    assert "carried over from training (not tunable; no fit)" in captured.out
    assert "carried over from training without a fit" in captured.err


# =========================== Fix 2: cgroup limits ===============================


def cgroup_tree(root: Path, levels: dict[str, dict[str, str]]) -> Path:
    """Write a fake unified hierarchy and a /proc/self/cgroup pointing at its leaf."""
    leaf = ""
    for rel, files in levels.items():
        node = root / rel if rel else root
        node.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (node / name).write_text(text + "\n", encoding="utf-8")
        leaf = rel
    proc = root.parent / "proc_cgroup"
    proc.write_text(f"0::/{leaf}\n", encoding="utf-8")
    return proc


def test_detects_the_tightest_limit_from_the_leaf_upwards(tmp_path: Path) -> None:
    root = tmp_path / "cg"
    proc = cgroup_tree(root, {
        "": {},
        "user.slice": {"memory.max": str(6 * GIB), "cpu.max": "max 100000"},
        "user.slice/run.scope": {"memory.max": str(8 * GIB), "cpu.max": "150000 100000"},
    })
    limits = detect_cgroup_limits(root=root, proc_cgroup=proc)
    assert limits.memory_bytes == 6 * GIB  # the ancestor binds tighter than the scope
    assert limits.cpus == 1.5


def test_unlimited_missing_and_cgroup_v1_mean_no_limit(tmp_path: Path) -> None:
    root = tmp_path / "cg"
    proc = cgroup_tree(root, {"": {}, "a.scope": {"memory.max": "max"}})  # no cpu.max
    assert detect_cgroup_limits(root=root, proc_cgroup=proc) == CgroupLimits(None, None)
    v1 = tmp_path / "v1"
    v1.write_text("12:memory:/user.slice\n", encoding="utf-8")
    assert detect_cgroup_limits(root=root, proc_cgroup=v1) == CgroupLimits(None, None)
    assert detect_cgroup_limits(root=root, proc_cgroup=tmp_path / "absent") == CgroupLimits(
        None, None
    )


def test_a_cgroup_limit_below_the_host_is_passed_on() -> None:
    limits = resolve_resource_limits(
        TrainingConfig(), cgroup=CgroupLimits(4 * GIB, 2.5), host_memory=16 * GIB, host_cpus=12
    )
    assert limits.fit_kwargs() == {"memory_limit": 4.0, "num_cpus": 2}
    assert (limits.memory_source, limits.cpu_source) == ("cgroup", "cgroup")
    assert limits.as_record()["host_memory_gb"] == 16.0


def test_a_cgroup_limit_at_or_above_the_host_is_left_to_autogluon() -> None:
    limits = resolve_resource_limits(
        TrainingConfig(), cgroup=CgroupLimits(16 * GIB, 12.0), host_memory=16 * GIB, host_cpus=12
    )
    assert limits.fit_kwargs() == {} and limits.memory_source is None


def test_a_fractional_cpu_quota_never_rounds_to_zero() -> None:
    limits = resolve_resource_limits(
        TrainingConfig(), cgroup=CgroupLimits(None, 0.5), host_memory=16 * GIB, host_cpus=12
    )
    assert limits.fit_kwargs() == {"num_cpus": 1}


def test_config_overrides_win_over_detection() -> None:
    limits = resolve_resource_limits(
        TrainingConfig(memory_limit_gb=3.5, num_cpus=4),
        cgroup=CgroupLimits(8 * GIB, 6.0), host_memory=16 * GIB, host_cpus=12,
    )
    assert limits.fit_kwargs() == {"memory_limit": 3.5, "num_cpus": 4}
    assert (limits.memory_source, limits.cpu_source) == ("config", "config")


def with_training(tmp_path: Path, training: Any) -> Any:
    raw = project_config(training=training)
    return load_config(write_project(tmp_path, raw=raw))


def test_training_resource_overrides_parse(tmp_path: Path) -> None:
    config = with_training(tmp_path, {"memory_limit_gb": 6, "num_cpus": 3})
    assert config.training.memory_limit_gb == 6.0 and config.training.num_cpus == 3
    assert isinstance(config.training.memory_limit_gb, float)


@pytest.mark.parametrize(
    ("training", "message"),
    [
        ({"memory_limit_gb": 0}, "memory_limit_gb"),
        ({"memory_limit_gb": -1.5}, "memory_limit_gb"),
        ({"memory_limit_gb": True}, "memory_limit_gb"),
        ({"memory_limit_gb": "8G"}, "memory_limit_gb"),
        ({"memory_limit_gb": float("inf")}, "memory_limit_gb"),
        ({"num_cpus": 0}, "num_cpus"),
        ({"num_cpus": 2.5}, "num_cpus"),
        ({"num_cpus": True}, "num_cpus"),
        ({"memory_limit": 4}, 'unsupported "training" setting'),
    ],
)
def test_invalid_resource_overrides_are_rejected(
    tmp_path: Path, training: Any, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        with_training(tmp_path, training)


class RaisingPredictor(FakePredictor):
    """Captures fit kwargs, then stops: enough to see what AutoGluon would receive."""

    def fit(self, **kwargs: Any) -> "RaisingPredictor":
        type(self).fit_kwargs = kwargs
        raise RuntimeError("captured")


@pytest.fixture
def four_gb_cgroup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resources, "detect_cgroup_limits", lambda *a, **k: CgroupLimits(4 * GIB, 2.0)
    )
    monkeypatch.setattr(resources, "host_memory_bytes", lambda: 16 * GIB)
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 12)


def fit_kwargs_of(call: str, tmp_path: Path) -> dict[str, Any]:
    import pandas as pd

    adapter = AutoGluonAdapter(predictor_factory=RaisingPredictor, version_resolver=lambda: "t")
    common = dict(
        task=TaskConfig("binary", "label", "yes"),
        model=ModelConfig("gbm", "GBM", {}),
        primary_metric="roc_auc",
        predictor_path=tmp_path / "p",
        training=TrainingConfig(seed=1),
    )
    train = pd.DataFrame({"x": [1, 2], "label": ["no", "yes"]})
    with pytest.raises(AutoGluonError, match="captured"):
        if call == "fit_predict":
            adapter.fit_predict(train_data=train, validation_features=train[["x"]], **common)
        else:
            adapter.fit_final(
                train_data=train, test_features=train[["x"]],
                best_hyperparameters={}, effective_seed=1, **common,
            )
    return RaisingPredictor.fit_kwargs


@pytest.mark.parametrize("call", ["fit_predict", "fit_final"])
def test_adapter_passes_the_limit_to_autogluon(
    tmp_path: Path, call: str, four_gb_cgroup: None
) -> None:
    kwargs = fit_kwargs_of(call, tmp_path)
    assert kwargs["memory_limit"] == 4.0 and kwargs["num_cpus"] == 2


@pytest.mark.parametrize("call", ["fit_predict", "fit_final"])
def test_adapter_passes_nothing_without_a_limit(tmp_path: Path, call: str) -> None:
    kwargs = fit_kwargs_of(call, tmp_path)
    assert "memory_limit" not in kwargs and "num_cpus" not in kwargs


def test_every_manifest_records_the_effective_limits(tmp_path: Path, four_gb_cgroup: None) -> None:
    build(tmp_path, through="finalize")
    expected = {
        "memory_limit_gb": 4.0, "memory_source": "cgroup", "num_cpus": 2, "cpu_source": "cgroup",
        "host_memory_gb": 16.0, "host_cpus": 12, "cgroup_memory_gb": 4.0, "cgroup_cpus": 2.0,
    }
    for phase in ("training", "tuning", "final"):
        manifest = json.loads((tmp_path / f".mltool/{phase}/manifest.json").read_text())
        assert manifest["resource_limits"] == expected, phase
