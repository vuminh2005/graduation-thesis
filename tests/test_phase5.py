from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import yaml

from mltool.autogluon_adapter import AutoGluonAdapter, AutoGluonError, AutoGluonOutput
from mltool.cli import (
    features_project,
    leaderboard_project,
    tune_project,
    tuning_leaderboard_project,
    validate_project,
)
from mltool.config import (
    ConfigError,
    HpoConfig,
    ModelConfig,
    TaskConfig,
    TrainingConfig,
    load_config,
)
from mltool.experiment import ExperimentError, build_experiment_plan
from mltool.training import TrainingError, train_experiment
from mltool.tuning import (
    TuningError,
    build_selected_configuration,
    build_tuning_selection,
    load_persisted_tuning,
    select_top_rows,
    tune_experiment,
)
from test_phase4 import FakePredictor, RecordingAdapter, materialize, project_config, write_project


MODELS = [
    {"name": "lightgbm", "family": "GBM", "params": {}},
    {"name": "forest", "family": "RF", "params": {}},
    {"name": "lgb2", "family": "GBM", "params": {"num_boost_round": 2}},
]


def hpo_raw(path: Path, **hpo: Any) -> None:
    raw = yaml.safe_load(path.read_text())
    raw["hpo"] = {"top_n": 2, "num_trials": 4, "time_limit_seconds": 30, **hpo}
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def trained_project(root: Path, **hpo: Any) -> Path:
    raw = project_config(models=MODELS)
    path = materialize(root, raw=raw)
    hpo_raw(path, **hpo)
    RecordingAdapter.failures = set()
    RecordingAdapter.calls = []
    train_experiment(build_experiment_plan(load_config(path)), adapter_factory=RecordingAdapter)
    return path


class HpoFakePredictor(FakePredictor):
    """FakePredictor subclass so per-test overrides never leak into Phase 4 tests."""

    model_best = "M"
    names = ["LightGBM"]
    hyperparameters: dict[str, Any] = {}

    def model_names(self) -> list[str]:
        return list(type(self).names)

    def info(self) -> dict[str, Any]:
        return {"model_info": {type(self).model_best: {"hyperparameters": type(self).hyperparameters}}}


class TuningAdapter(RecordingAdapter):
    """Recording adapter that also emulates HPO output fields."""

    calls: list[dict[str, Any]] = []
    failures: set[str] = set()
    boost: dict[str, float] = {}

    def fit_predict(self, **kwargs: Any) -> AutoGluonOutput:
        type(self).calls.append(kwargs)
        output = RecordingAdapter().fit_predict(**{**kwargs, "hpo": None})
        RecordingAdapter.calls.pop()
        if kwargs["model"].name in type(self).failures:
            raise AutoGluonError("hpo exploded")
        output.best_model = "LightGBM/T3"
        output.best_hyperparameters = {"learning_rate": 0.05, "seed": kwargs["model"].name}
        output.trained_models = [output.trained_models[0] + "/T1", output.trained_models[0] + "/T2"]
        return output


# --- config ----------------------------------------------------------------


def test_hpo_config_defaults_and_required_time_limit(tmp_path: Path) -> None:
    path = write_project(tmp_path, raw=project_config(models=MODELS))
    assert load_config(path).hpo is None
    raw = yaml.safe_load(path.read_text())
    raw["hpo"] = {"time_limit_seconds": 60}
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path).hpo == HpoConfig(top_n=3, num_trials=10, time_limit_seconds=60)
    for bad in ({}, {"time_limit_seconds": None}, {"time_limit_seconds": 0},
                {"time_limit_seconds": 5, "top_n": 0}, {"time_limit_seconds": 5, "num_trials": True},
                {"time_limit_seconds": 5, "gpus": 1}):
        raw["hpo"] = bad
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ConfigError, match="hpo"):
            load_config(path)


# --- selection ---------------------------------------------------------------


def rows(*scores: tuple[str, float | None, str]) -> list[dict[str, Any]]:
    return [
        {"candidate_id": name, "primary_score": score, "status": status}
        for name, score, status in scores
    ]


def test_top_n_selection_respects_direction_and_status() -> None:
    data = rows(("a", 0.5, "SUCCEEDED"), ("b", 0.9, "SUCCEEDED"), ("c", 0.7, "SUCCEEDED"),
                ("d", None, "FAILED"))
    picked, warning = select_top_rows(data, top_n=2, direction="maximize")
    assert [row["candidate_id"] for row in picked] == ["b", "c"] and warning is None
    picked, _ = select_top_rows(data, top_n=2, direction="minimize")
    assert [row["candidate_id"] for row in picked] == ["a", "c"]


def test_fewer_succeeded_than_top_n_warns_but_does_not_fail() -> None:
    picked, warning = select_top_rows(
        rows(("a", 0.5, "SUCCEEDED"), ("d", None, "FAILED")), top_n=3, direction="maximize"
    )
    assert [row["candidate_id"] for row in picked] == ["a"]
    assert warning and "only 1" in warning


def test_selection_uses_persisted_training_leaderboard(tmp_path: Path) -> None:
    path = trained_project(tmp_path, top_n=2)
    selection = build_tuning_selection(build_experiment_plan(load_config(path)))
    assert len(selection.candidates) == 2
    assert selection.warning is None
    path = trained_project(tmp_path / "wide", top_n=9)
    selection = build_tuning_selection(build_experiment_plan(load_config(path)))
    assert len(selection.candidates) == 3 and selection.warning


# --- adapter -----------------------------------------------------------------


def adapter_call(hpo: HpoConfig | None, tmp_path: Path) -> dict[str, Any]:
    adapter = AutoGluonAdapter(predictor_factory=HpoFakePredictor, version_resolver=lambda: "t")
    HpoFakePredictor.validation_frames = []
    HpoFakePredictor.model_best = "LightGBM/T2"
    HpoFakePredictor.hyperparameters = {"learning_rate": 0.1}
    output = adapter.fit_predict(
        train_data=pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]}),
        validation_features=pd.DataFrame({"x": [4, 5]}),
        task=TaskConfig("binary", "label", "yes"),
        model=ModelConfig("gbm", "GBM", {}),
        primary_metric="roc_auc",
        predictor_path=tmp_path / "p",
        training=TrainingConfig(9),
        hpo=hpo,
    )
    return {"fit": HpoFakePredictor.fit_kwargs, "output": output}


def test_adapter_hpo_enables_tuning_and_keeps_phase4_guards(tmp_path: Path) -> None:
    result = adapter_call(HpoConfig(2, 7, 41), tmp_path)
    fit = result["fit"]
    assert fit["hyperparameter_tune_kwargs"] == {
        "num_trials": 7, "scheduler": "local", "searcher": "random"
    }
    assert fit["time_limit"] == 41  # hpo budget overrides training.time_limit_seconds
    assert "tuning_data" not in fit and "validation_features" not in fit
    assert fit["num_bag_folds"] == 0 and fit["num_stack_levels"] == 0
    assert fit["dynamic_stacking"] is False
    assert fit["fit_weighted_ensemble"] is False
    assert fit["fit_full_last_level_weighted_ensemble"] is False
    assert fit["full_weighted_ensemble_additionally"] is False
    assert fit["num_gpus"] == 0 and fit["fit_strategy"] == "sequential"
    assert result["output"].best_hyperparameters == {"learning_rate": 0.1}
    assert result["output"].best_model == "LightGBM/T2"


def test_adapter_without_hpo_is_unchanged_phase4(tmp_path: Path) -> None:
    fit = adapter_call(None, tmp_path)["fit"]
    assert fit["hyperparameter_tune_kwargs"] is None and fit["time_limit"] == 9


# --- tune --------------------------------------------------------------------


def test_tune_persists_artifacts_selects_best_and_never_reads_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = trained_project(tmp_path)
    plan = build_experiment_plan(load_config(path))
    selection = build_tuning_selection(plan)
    import mltool.training as training_module

    real_read = training_module.pd.read_parquet

    def guarded(path_arg: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        assert Path(path_arg).name != "test.parquet"
        return real_read(path_arg, *args, **kwargs)

    monkeypatch.setattr(training_module.pd, "read_parquet", guarded)
    TuningAdapter.calls = []
    TuningAdapter.failures = set()
    result = tune_experiment(plan, selection, adapter_factory=TuningAdapter)

    assert len(TuningAdapter.calls) == 2  # top_n
    for call in TuningAdapter.calls:
        assert call["hpo"] == HpoConfig(2, 4, 30)
        assert "test" not in str(call["predictor_path"].name)
        assert "label" not in call["validation_features"]
        assert ".tuning-staging-" in str(call["predictor_path"])  # staged first
    out = tmp_path / ".mltool/tuning"
    assert not list((tmp_path / ".mltool").glob(".tuning-staging-*"))
    for candidate in selection.candidates:
        candidate_dir = out / "candidates" / candidate.candidate_id
        stored = json.loads((candidate_dir / "result.json").read_text())
        assert stored["best_hyperparameters"]["learning_rate"] == 0.05
        assert stored["hpo"]["num_trials"] == 4
        assert (candidate_dir / "predictor/fake.txt").is_file()
        assert stored["predictor_path"] == str(candidate_dir / "predictor")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["test_data_used"] is False
    assert manifest["hpo"]["time_limit_seconds"] == 30
    assert (out / "leaderboard.csv").is_file() and (out / "leaderboard.json").is_file()
    selected = json.loads((out / "selected.json").read_text())
    assert selected["candidate_id"] == result.leaderboard[0]["candidate_id"]
    assert selected["candidate_id"] == manifest["selected_candidate_id"]
    assert selected["best_hyperparameters"]["learning_rate"] == 0.05
    assert selected["tuned_validation_score"] == result.leaderboard[0]["primary_score"]
    assert selected["test_data_used"] is False


def test_selected_json_is_the_best_tuned_candidate() -> None:
    def result(name: str, score: float) -> dict[str, Any]:
        return {
            "candidate_id": name, "feature_set": "base",
            "model": {"name": name, "family": "GBM", "params": {}},
            "best_hyperparameters": {"k": name}, "best_model": "m",
            "metrics": {"rmse": score}, "phase4_primary_score": 9.0,
            "feature_artifacts": {}, "positive_class": None,
            "autogluon_version": "x", "predictor_path": "p", "status": "SUCCEEDED",
            "training_seconds": 1.0, "hpo_effective": True, "seed": None,
        }

    from mltool.training import build_leaderboard

    results = [result("a", 3.0), result("b", 1.0), result("c", 2.0)]
    board = build_leaderboard(results, primary_metric="rmse", direction="minimize")
    selected = build_selected_configuration(board, results, primary_metric="rmse")
    assert selected and selected["candidate_id"] == "b"
    assert selected["best_hyperparameters"] == {"k": "b"}
    assert build_selected_configuration([], [], primary_metric="rmse") is None


def test_failed_tuning_candidate_is_recorded_and_others_continue(tmp_path: Path) -> None:
    path = trained_project(tmp_path)
    plan = build_experiment_plan(load_config(path))
    selection = build_tuning_selection(plan)
    TuningAdapter.calls = []
    TuningAdapter.failures = {selection.candidates[0].model.name}
    result = tune_experiment(plan, selection, adapter_factory=TuningAdapter)
    assert [r["status"] for r in result.candidate_results] == ["FAILED", "SUCCEEDED"]
    assert result.selected and result.selected["candidate_id"] == selection.candidates[1].candidate_id
    TuningAdapter.failures = {c.model.name for c in selection.candidates}
    result = tune_experiment(plan, selection, adapter_factory=TuningAdapter)
    assert not result.is_successful and result.selected is None
    assert not (tmp_path / ".mltool/tuning/selected.json").exists()


def test_rerun_replaces_tuning_directory_atomically(tmp_path: Path) -> None:
    path = trained_project(tmp_path)
    plan = build_experiment_plan(load_config(path))
    selection = build_tuning_selection(plan)
    TuningAdapter.failures = set()
    tune_experiment(plan, selection, adapter_factory=TuningAdapter)
    tune_experiment(plan, selection, adapter_factory=TuningAdapter)
    assert not [p for p in (tmp_path / ".mltool").iterdir() if p.name.startswith(".tuning-")]


# --- errors ------------------------------------------------------------------


def test_tune_requires_hpo_section(tmp_path: Path) -> None:
    path = materialize(tmp_path, raw=project_config(models=MODELS))
    with pytest.raises(TuningError, match='"hpo" section'):
        build_tuning_selection(build_experiment_plan(load_config(path)))
    assert tune_project(path) == 2


def test_tune_errors_clearly_when_training_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = materialize(tmp_path, raw=project_config(models=MODELS))
    hpo_raw(path)
    with pytest.raises(TrainingError, match='run "mltool train" first'):
        build_tuning_selection(build_experiment_plan(load_config(path)))
    assert tune_project(path) == 2
    assert 'run "mltool train" first' in capsys.readouterr().out
    assert not (tmp_path / ".mltool/tuning").exists()


def test_tune_refuses_stale_training_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "mltool.cli.tune_experiment", lambda *a, **k: pytest.fail("must not tune on stale data")
    )
    # model config changed after training
    path = trained_project(tmp_path / "models")
    raw = yaml.safe_load(path.read_text())
    raw["models"][0]["params"] = {"num_boost_round": 5}
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert tune_project(path) == 2
    assert "training artifacts are stale" in capsys.readouterr().out
    # features re-materialized after training (hash change)
    path = trained_project(tmp_path / "features")
    frame = pd.read_parquet(tmp_path / "features/.mltool/features/base/train.parquet")
    frame.iloc[0, 0] = frame.iloc[0, 0] + 1
    frame.to_parquet(tmp_path / "features/.mltool/features/base/train.parquet", index=False)
    assert tune_project(path) == 2
    assert "stale" in capsys.readouterr().out
    # feature artifacts stale relative to the dataset
    path = trained_project(tmp_path / "data")
    (tmp_path / "data/data/dataset.csv").write_text("x,z,label\n1,1,0\n2,2,1\n")
    assert tune_project(path) == 2
    assert "stale" in capsys.readouterr().out


def test_tuning_leaderboard_is_read_only_and_warns_on_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = trained_project(tmp_path)
    assert tuning_leaderboard_project(path) == 2
    plan = build_experiment_plan(load_config(path))
    TuningAdapter.failures = set()
    tune_experiment(plan, build_tuning_selection(plan), adapter_factory=TuningAdapter)
    before = (tmp_path / ".mltool/tuning/leaderboard.json").read_bytes()
    monkeypatch.setattr(
        "mltool.cli.tune_experiment", lambda *a, **k: pytest.fail("must not re-tune")
    )
    assert tuning_leaderboard_project(path) == 0
    out = capsys.readouterr().out
    assert "MLTool tuning leaderboard" in out and "Selected:" in out
    assert (tmp_path / ".mltool/tuning/leaderboard.json").read_bytes() == before
    assert load_persisted_tuning(load_config(path)).warning is None
    hpo_raw(path, num_trials=5)
    assert load_persisted_tuning(load_config(path)).warning is not None


def test_tune_cli_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = trained_project(tmp_path)
    monkeypatch.setattr("mltool.cli.tune_experiment", lambda p, s: SimpleNamespace(is_successful=True, render=lambda: "ok"))
    assert tune_project(path) == 0
    monkeypatch.setattr("mltool.cli.tune_experiment", lambda p, s: SimpleNamespace(is_successful=False, render=lambda: "no"))
    assert tune_project(path) == 1


def test_phase4_commands_still_work_and_tuning_sources_never_mention_test_artifact() -> None:
    import mltool.tuning as tuning_module

    assert "test.parquet" not in Path(tuning_module.__file__).read_text()


# --- seed ------------------------------------------------------------------


def seeded_call(tmp_path: Path, family: str, params: dict[str, Any], seed: int | None,
                hpo: HpoConfig | None) -> tuple[dict[str, Any], dict[str, Any]]:
    adapter = AutoGluonAdapter(predictor_factory=HpoFakePredictor, version_resolver=lambda: "t")
    HpoFakePredictor.model_best = "M"
    HpoFakePredictor.hyperparameters = {}
    token = {"GBM": "LightGBM", "RF": "RandomForest", "CAT": "CatBoost",
             "XGB": "XGBoost", "XT": "ExtraTrees"}[family]
    HpoFakePredictor.names = [token]
    adapter.fit_predict(
        train_data=pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]}),
        validation_features=pd.DataFrame({"x": [4, 5]}),
        task=TaskConfig("binary", "label", "yes"),
        model=ModelConfig("m", family, params),
        primary_metric="roc_auc",
        predictor_path=tmp_path / "p",
        training=TrainingConfig(None, seed),
        hpo=hpo,
    )
    return HpoFakePredictor.init_kwargs, HpoFakePredictor.fit_kwargs


@pytest.mark.parametrize("hpo", [None, HpoConfig(1, 3, 20)])
@pytest.mark.parametrize(
    ("family", "key"),
    [("GBM", "seed"), ("RF", "random_state"), ("XT", "random_state"),
     ("CAT", "random_seed"), ("XGB", "seed")],
)
def test_seed_reaches_fit_in_both_phase4_and_phase5_paths(
    tmp_path: Path, family: str, key: str, hpo: HpoConfig | None
) -> None:
    init, fit = seeded_call(tmp_path, family, {"depth": 2}, 123, hpo)
    assert fit["hyperparameters"] == {family: {"depth": 2, key: 123}}
    assert init["learner_kwargs"] == {"random_state": 123}


def test_explicit_model_seed_wins_and_unset_seed_changes_nothing(tmp_path: Path) -> None:
    _, fit = seeded_call(tmp_path, "GBM", {"seed": 7}, 123, None)
    assert fit["hyperparameters"] == {"GBM": {"seed": 7}}
    init, fit = seeded_call(tmp_path, "GBM", {}, None, None)
    assert fit["hyperparameters"] == {"GBM": {}} and "learner_kwargs" not in init


def test_training_seed_config_validation(tmp_path: Path) -> None:
    path = write_project(tmp_path, raw=project_config(models=MODELS, training={"seed": 5}))
    assert load_config(path).training.seed == 5
    for bad in (True, 1.5, -1, "a", 2**31):
        raw = yaml.safe_load(path.read_text())
        raw["training"]["seed"] = bad
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ConfigError, match="training.seed"):
            load_config(path)


# --- families without a search space -----------------------------------------


def test_rf_candidates_are_flagged_as_not_tuned(tmp_path: Path) -> None:
    path = trained_project(tmp_path, top_n=3)
    plan = build_experiment_plan(load_config(path))
    selection = build_tuning_selection(plan)
    TuningAdapter.failures = set()
    result = tune_experiment(plan, selection, adapter_factory=TuningAdapter)
    by_family = {r["model"]["family"]: r for r in result.candidate_results}
    assert by_family["RF"]["hpo_effective"] is False
    assert "no default search space for family RF" in by_family["RF"]["hpo_warning"]
    assert by_family["GBM"]["hpo_effective"] is True and by_family["GBM"]["hpo_warning"] is None
    report = result.render()
    assert "HPO has no default search space for family RF" in report
    assert "not tuned" in report and "family GBM" not in report
    stored = json.loads(
        (tmp_path / ".mltool/tuning/candidates/base__forest/result.json").read_text()
    )
    assert stored["hpo_warning"] and stored["hpo_effective"] is False
