"""Phase 8: AutoGluon ENSEMBLE as one more FeatureSet x model candidate.

Fake-predictor style, matching test_phase4/5/6/7. Nothing here touches real
AutoGluon; the real smoke test lives in the Phase-8 report, not in this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from mltool.autogluon_adapter import (
    AutoGluonAdapter,
    AutoGluonError,
    AutoGluonOutput,
    _check_ensemble_trained_models,
    _check_trained_models,
    effective_model_seed,
)
from mltool.cli import (
    best_project,
    final_result_project,
    finalize_project,
    register_project,
    train_project,
    tune_project,
)
from mltool.config import ConfigError, ModelConfig, TaskConfig, TrainingConfig, load_config
from mltool.experiment import build_experiment_plan
from mltool.finalize import finalize_experiment, load_finalize_input
from mltool.registry import register_final
from mltool.training import train_experiment
from mltool.tuning import (
    build_tuning_selection,
    ensemble_hpo_not_applied_warning,
    hpo_not_applicable_warning,
    tune_experiment,
)
from test_phase4 import FakePredictor, RecordingAdapter, materialize, project_config, write_project
from test_phase5 import HpoFakePredictor, hpo_raw
from test_phase6 import FinalFakePredictor

TOKEN = {"GBM": "LightGBM", "CAT": "CatBoost", "XGB": "XGBoost", "RF": "RandomForest", "XT": "ExtraTrees"}

ENSEMBLE_DEFAULT = {"name": "ag_ensemble", "family": "ENSEMBLE", "params": {}}
GBM = {"name": "gbm", "family": "GBM", "params": {}}
RF = {"name": "rf", "family": "RF", "params": {}}


def ensemble_model(**params: Any) -> dict[str, Any]:
    return {"name": "ag_ensemble", "family": "ENSEMBLE", "params": params}


# =========================== Config validation ================================


def test_ensemble_defaults(tmp_path: Path) -> None:
    path = write_project(tmp_path, raw=project_config(models=[ENSEMBLE_DEFAULT]))
    config = load_config(path)
    assert config.models == [
        ModelConfig(
            "ag_ensemble",
            "ENSEMBLE",
            {"families": ["GBM", "CAT", "XGB", "RF", "XT"], "num_bag_folds": 3, "num_stack_levels": 1},
        )
    ]


def test_ensemble_explicit_params_are_preserved(tmp_path: Path) -> None:
    raw = project_config(
        models=[ensemble_model(families=["GBM", "RF"], num_bag_folds=5, num_stack_levels=0)]
    )
    config = load_config(write_project(tmp_path, raw=raw))
    assert config.models[0].params == {
        "families": ["GBM", "RF"], "num_bag_folds": 5, "num_stack_levels": 0
    }


def test_ensemble_num_bag_folds_zero_is_allowed(tmp_path: Path) -> None:
    raw = project_config(models=[ensemble_model(num_bag_folds=0, num_stack_levels=0)])
    config = load_config(write_project(tmp_path, raw=raw))
    assert config.models[0].params["num_bag_folds"] == 0


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"families": []}, "non-empty list"),
        ({"families": ["GBM", "GBM"]}, "unique"),
        ({"families": ["NN"]}, "unsupported family"),
        ({"families": ["ENSEMBLE"]}, "unsupported family"),
        ({"families": [1]}, "list of strings"),
        ({"num_bag_folds": 1}, "0 or an integer >= 2"),
        ({"num_bag_folds": -1}, "0 or an integer >= 2"),
        ({"num_bag_folds": True}, "0 or an integer >= 2"),
        ({"num_bag_folds": 3.5}, "0 or an integer >= 2"),
        ({"num_stack_levels": -1}, "non-negative integer"),
        ({"num_stack_levels": True}, "non-negative integer"),
        ({"num_bag_folds": 0, "num_stack_levels": 1}, "requires"),
        ({"num_bag_folds": 1, "num_stack_levels": 1}, "0 or an integer >= 2"),
        ({"trials": 5}, "unsupported"),
        ({"time_limit_seconds": 5}, "unsupported"),
    ],
)
def test_invalid_ensemble_params_are_rejected(
    tmp_path: Path, params: dict[str, Any], message: str
) -> None:
    raw = project_config(models=[ensemble_model(**params)])
    with pytest.raises(ConfigError, match=message):
        load_config(write_project(tmp_path, raw=raw))


def test_ensemble_family_is_accepted_alongside_single_families(tmp_path: Path) -> None:
    raw = project_config(models=[GBM, RF, ENSEMBLE_DEFAULT])
    config = load_config(write_project(tmp_path, raw=raw))
    assert [model.family for model in config.models] == ["GBM", "RF", "ENSEMBLE"]


def test_multiple_ensemble_entries_with_different_params_are_allowed(tmp_path: Path) -> None:
    raw = project_config(
        models=[
            {"name": "e1", "family": "ENSEMBLE", "params": {"families": ["GBM", "RF"]}},
            {"name": "e2", "family": "ENSEMBLE", "params": {"num_bag_folds": 4}},
        ]
    )
    config = load_config(write_project(tmp_path, raw=raw))
    assert len(config.models) == 2
    assert config.models[0].params["families"] == ["GBM", "RF"]
    assert config.models[1].params["num_bag_folds"] == 4


# =========================== Adapter: fit kwargs ===============================


def test_non_ensemble_fit_predict_kwargs_are_byte_for_byte_unchanged(tmp_path: Path) -> None:
    """Regression guard: Phase 8 must not touch the single-family fit path."""
    FakePredictor.validation_frames = []
    adapter = AutoGluonAdapter(predictor_factory=FakePredictor, version_resolver=lambda: "1.6.test")
    adapter.fit_predict(
        train_data=pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]}),
        validation_features=pd.DataFrame({"x": [4, 5]}),
        task=TaskConfig("binary", "label", "yes"),
        model=ModelConfig("gbm", "GBM", {"num_boost_round": 7}),
        primary_metric="roc_auc",
        predictor_path=tmp_path / "predictor",
        training=TrainingConfig(9),
    )
    fit = FakePredictor.fit_kwargs
    assert fit == {
        "train_data": fit["train_data"],  # a DataFrame; equality checked below
        "hyperparameters": {"GBM": {"num_boost_round": 7}},
        "hyperparameter_tune_kwargs": None,
        "fit_weighted_ensemble": False,
        "fit_full_last_level_weighted_ensemble": False,
        "full_weighted_ensemble_additionally": False,
        "num_bag_folds": 0,
        "num_stack_levels": 0,
        "dynamic_stacking": False,
        "num_gpus": 0,
        "fit_strategy": "sequential",
        "time_limit": 9,
    }
    assert "ag_args_ensemble" not in fit


def ensemble_predictor_factory(names: list[str], best: str, *, final: bool = False) -> type:
    base = FinalFakePredictor if final else HpoFakePredictor

    class _Ensemble(base):  # type: ignore[misc,valid-type]
        pass

    _Ensemble.names = names
    _Ensemble.model_best = best
    return _Ensemble


def test_ensemble_fit_predict_kwargs(tmp_path: Path) -> None:
    factory = ensemble_predictor_factory(
        ["LightGBM_BAG_L1", "CatBoost_BAG_L1", "WeightedEnsemble_L2"], "WeightedEnsemble_L2"
    )
    adapter = AutoGluonAdapter(predictor_factory=factory, version_resolver=lambda: "t")
    model = ModelConfig(
        "ag_ensemble", "ENSEMBLE",
        {"families": ["GBM", "CAT"], "num_bag_folds": 4, "num_stack_levels": 2},
    )
    output = adapter.fit_predict(
        train_data=pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]}),
        validation_features=pd.DataFrame({"x": [4, 5]}),
        task=TaskConfig("binary", "label", "yes"),
        model=model,
        primary_metric="roc_auc",
        predictor_path=tmp_path / "p",
        training=TrainingConfig(time_limit_seconds=None, seed=7),
    )
    fit = factory.fit_kwargs
    assert fit["hyperparameters"] == {"GBM": {"seed": 7}, "CAT": {"random_seed": 7}}
    assert fit["num_bag_folds"] == 4 and fit["num_stack_levels"] == 2
    assert fit["fit_weighted_ensemble"] is True
    assert fit["ag_args_ensemble"] == {"fold_fitting_strategy": "sequential_local"}
    assert fit["hyperparameter_tune_kwargs"] is None
    assert fit["dynamic_stacking"] is False
    assert fit["num_gpus"] == 0 and fit["fit_strategy"] == "sequential"
    assert "tuning_data" not in fit and "validation_features" not in fit
    assert factory.init_kwargs["learner_kwargs"] == {"random_state": 7}
    assert output.best_model == "WeightedEnsemble_L2"
    assert output.best_hyperparameters == {
        "families": ["GBM", "CAT"], "num_bag_folds": 4, "num_stack_levels": 2
    }


def test_ensemble_num_bag_folds_zero_is_passed_through(tmp_path: Path) -> None:
    factory = ensemble_predictor_factory(["LightGBM", "WeightedEnsemble_L2"], "WeightedEnsemble_L2")
    adapter = AutoGluonAdapter(predictor_factory=factory, version_resolver=lambda: "t")
    model = ModelConfig(
        "ag_ensemble", "ENSEMBLE", {"families": ["GBM"], "num_bag_folds": 0, "num_stack_levels": 0}
    )
    adapter.fit_predict(
        train_data=pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]}),
        validation_features=pd.DataFrame({"x": [4, 5]}),
        task=TaskConfig("binary", "label", "yes"),
        model=model,
        primary_metric="roc_auc",
        predictor_path=tmp_path / "p",
        training=TrainingConfig(),
    )
    assert factory.fit_kwargs["num_bag_folds"] == 0 and factory.fit_kwargs["num_stack_levels"] == 0


def test_ensemble_fit_predict_ignores_no_seed(tmp_path: Path) -> None:
    factory = ensemble_predictor_factory(["LightGBM", "WeightedEnsemble_L2"], "WeightedEnsemble_L2")
    adapter = AutoGluonAdapter(predictor_factory=factory, version_resolver=lambda: "t")
    model = ModelConfig(
        "ag_ensemble", "ENSEMBLE", {"families": ["GBM"], "num_bag_folds": 2, "num_stack_levels": 0}
    )
    adapter.fit_predict(
        train_data=pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]}),
        validation_features=pd.DataFrame({"x": [4, 5]}),
        task=TaskConfig("binary", "label", "yes"),
        model=model,
        primary_metric="roc_auc",
        predictor_path=tmp_path / "p",
        training=TrainingConfig(),
    )
    assert factory.fit_kwargs["hyperparameters"] == {"GBM": {}}
    assert "learner_kwargs" not in factory.init_kwargs


def test_ensemble_fit_final_kwargs_and_refit(tmp_path: Path) -> None:
    factory = ensemble_predictor_factory(
        ["LightGBM_BAG_L1_FULL", "RandomForest_BAG_L1_FULL", "WeightedEnsemble_L2_FULL"],
        "WeightedEnsemble_L2_FULL",
        final=True,
    )
    adapter = AutoGluonAdapter(predictor_factory=factory, version_resolver=lambda: "t")
    output = adapter.fit_final(
        train_data=pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]}),
        test_features=pd.DataFrame({"x": [4, 5]}),
        task=TaskConfig("binary", "label", "yes"),
        model=ModelConfig("ag_ensemble", "ENSEMBLE", {}),
        best_hyperparameters={"families": ["GBM", "RF"], "num_bag_folds": 3, "num_stack_levels": 1},
        effective_seed=13,
        primary_metric="roc_auc",
        predictor_path=tmp_path / "p",
        training=TrainingConfig(),
    )
    fit = factory.fit_kwargs
    assert fit["hyperparameters"] == {"GBM": {"seed": 13}, "RF": {"random_state": 13}}
    assert fit["num_bag_folds"] == 3 and fit["num_stack_levels"] == 1
    assert fit["fit_weighted_ensemble"] is True
    assert fit["ag_args_ensemble"] == {"fold_fitting_strategy": "sequential_local"}
    assert fit["refit_full"] is True and fit["set_best_to_refit_full"] is True
    assert fit["hyperparameter_tune_kwargs"] is None
    assert output.best_model == "WeightedEnsemble_L2_FULL"
    assert output.best_hyperparameters == {"GBM": {"seed": 13}, "RF": {"random_state": 13}}


def test_ensemble_fit_final_with_no_bagging_still_refits_full_with_time_limit(
    tmp_path: Path,
) -> None:
    """Fix 3: the num_bag_folds=0 path, confirmed for real on Titanic-shaped data;
    locked in here with the exact model-name shape AutoGluon actually produced."""
    factory = ensemble_predictor_factory(
        ["LightGBM", "RandomForest", "WeightedEnsemble_L2",
         "LightGBM_FULL", "RandomForest_FULL", "WeightedEnsemble_L2_FULL"],
        "WeightedEnsemble_L2_FULL",
        final=True,
    )
    adapter = AutoGluonAdapter(predictor_factory=factory, version_resolver=lambda: "t")
    output = adapter.fit_final(
        train_data=pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]}),
        test_features=pd.DataFrame({"x": [4, 5]}),
        task=TaskConfig("binary", "label", "yes"),
        model=ModelConfig("ag_ensemble", "ENSEMBLE", {}),
        best_hyperparameters={"families": ["GBM", "RF"], "num_bag_folds": 0, "num_stack_levels": 0},
        effective_seed=9,
        primary_metric="roc_auc",
        predictor_path=tmp_path / "p",
        training=TrainingConfig(time_limit_seconds=60),
    )
    fit = factory.fit_kwargs
    assert fit["num_bag_folds"] == 0 and fit["num_stack_levels"] == 0
    assert fit["time_limit"] == 60
    assert fit["refit_full"] is True and fit["set_best_to_refit_full"] is True
    assert output.best_model == "WeightedEnsemble_L2_FULL"


# =========================== trained-model guards ==============================


def test_check_trained_models_still_rejects_weighted_ensemble_for_single_family() -> None:
    with pytest.raises(AutoGluonError, match="weighted ensemble"):
        _check_trained_models(["LightGBM", "WeightedEnsemble_L2"], "GBM")
    with pytest.raises(AutoGluonError, match="outside family GBM"):
        _check_trained_models(["LightGBM", "RandomForest"], "GBM")
    _check_trained_models(["LightGBM"], "GBM")  # unchanged happy path


def test_check_ensemble_trained_models_accepts_members_and_weighted_ensemble() -> None:
    _check_ensemble_trained_models(
        ["LightGBM_BAG_L1", "CatBoost_BAG_L1", "WeightedEnsemble_L2", "LightGBM_BAG_L1_FULL"],
        ["GBM", "CAT"],
    )


def test_check_ensemble_trained_models_accepts_every_real_name_shape_observed() -> None:
    """Every shape seen across the Phase 8 real runs (Titanic, regression, multiclass)."""
    _check_ensemble_trained_models(
        [
            "LightGBM_BAG_L1", "RandomForest_BAG_L1", "CatBoost_BAG_L1", "ExtraTrees_BAG_L1",
            "XGBoost_BAG_L1", "WeightedEnsemble_L2", "LightGBM_BAG_L2", "RandomForest_BAG_L2",
            "CatBoost_BAG_L2", "ExtraTrees_BAG_L2", "XGBoost_BAG_L2", "WeightedEnsemble_L3",
        ],
        ["GBM", "CAT", "XGB", "RF", "XT"],
    )
    _check_ensemble_trained_models(  # finalize's refit_full, num_stack_levels=0
        ["LightGBM_BAG_L1", "RandomForest_BAG_L1", "WeightedEnsemble_L2",
         "LightGBM_BAG_L1_FULL", "RandomForest_BAG_L1_FULL", "WeightedEnsemble_L2_FULL"],
        ["GBM", "RF"],
    )
    _check_ensemble_trained_models(  # no bagging: num_bag_folds=0
        ["LightGBM", "RandomForest", "WeightedEnsemble_L2"], ["GBM", "RF"]
    )
    _check_ensemble_trained_models(  # no bagging, after refit_full
        ["LightGBM_FULL", "RandomForest_FULL", "WeightedEnsemble_L2_FULL"], ["GBM", "RF"]
    )


@pytest.mark.parametrize(
    "name",
    [
        "WeightedEnsembleX_L2",  # extra text: the old substring check let this through
        "WeightedEnsemble_L2_PARTIAL",  # near-miss suffix
        "WeightedEnsemble",  # missing the stack-level suffix entirely
        "LightGBMFake_BAG_L1",  # "LightGBM" only as a substring, not the real token
        "LightGBM_BAG_L1_2",  # malformed suffix, not "_FULL"
        "NotLightGBM_BAG_L1",  # token embedded, not a prefix match
        "LightGBM_BAG_LX",  # non-numeric fold level
    ],
)
def test_check_ensemble_trained_models_rejects_near_miss_names(name: str) -> None:
    with pytest.raises(AutoGluonError, match="outside the configured ensemble families"):
        _check_ensemble_trained_models(["LightGBM_BAG_L1", name], ["GBM"])


def test_check_ensemble_trained_models_rejects_a_foreign_family() -> None:
    with pytest.raises(AutoGluonError, match="outside the configured ensemble families"):
        _check_ensemble_trained_models(["LightGBM_BAG_L1", "XGBoost_BAG_L1"], ["GBM"])


def test_check_ensemble_trained_models_rejects_empty() -> None:
    with pytest.raises(AutoGluonError, match="did not train a usable model"):
        _check_ensemble_trained_models([], ["GBM"])


# =========================== Seed ==============================================


def test_effective_model_seed_for_ensemble_is_training_seed() -> None:
    assert effective_model_seed(
        ModelConfig("e", "ENSEMBLE", {"families": ["GBM"]}), TrainingConfig(None, 42)
    ) == 42
    assert effective_model_seed(
        ModelConfig("e", "ENSEMBLE", {"families": ["GBM"]}), TrainingConfig(None, None)
    ) is None


def test_seed_reaches_learner_and_every_member_family(tmp_path: Path) -> None:
    factory = ensemble_predictor_factory(
        [f"{TOKEN[f]}_BAG_L1" for f in ("GBM", "CAT", "XGB", "RF", "XT")] + ["WeightedEnsemble_L2"],
        "WeightedEnsemble_L2",
    )
    adapter = AutoGluonAdapter(predictor_factory=factory, version_resolver=lambda: "t")
    model = ModelConfig(
        "ag_ensemble", "ENSEMBLE",
        {"families": ["GBM", "CAT", "XGB", "RF", "XT"], "num_bag_folds": 2, "num_stack_levels": 0},
    )
    adapter.fit_predict(
        train_data=pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]}),
        validation_features=pd.DataFrame({"x": [4, 5]}),
        task=TaskConfig("binary", "label", "yes"),
        model=model,
        primary_metric="roc_auc",
        predictor_path=tmp_path / "p",
        training=TrainingConfig(time_limit_seconds=None, seed=21),
    )
    assert factory.init_kwargs["learner_kwargs"] == {"random_state": 21}
    hp = factory.fit_kwargs["hyperparameters"]
    assert hp == {
        "GBM": {"seed": 21}, "CAT": {"random_seed": 21}, "XGB": {"seed": 21},
        "RF": {"random_state": 21}, "XT": {"random_state": 21},
    }


# =========================== Full pipeline (train/tune/finalize) ===============


class EnsembleAwareAdapter:
    """Records calls; emulates AutoGluon's per-family/ensemble output generically
    enough to exercise train/tune/finalize wiring without real AutoGluon."""

    calls: list[dict[str, Any]] = []
    failures: set[str] = set()

    def fit_predict(self, **kwargs: Any) -> AutoGluonOutput:
        type(self).calls.append(kwargs)
        model = kwargs["model"]
        if model.name in type(self).failures:
            raise AutoGluonError(f"intentional failure for {model.name}")
        features = kwargs["validation_features"]
        task = kwargs["task"]
        if task.type == "regression":
            predictions = features["x"].astype(float)
            probabilities = None
            positive = None
        elif task.positive_class == "yes":
            predictions = features["x"].map(lambda value: "yes" if value % 2 else "no")
            yes = features["x"].map(lambda value: 0.9 if value % 2 else 0.1)
            probabilities = pd.DataFrame({"no": 1 - yes, "yes": yes}, index=features.index)
            positive = "yes"
        else:
            predictions = features["x"].map(lambda value: value % 2)
            one = features["x"].map(lambda value: 0.9 if value % 2 else 0.1)
            probabilities = pd.DataFrame({0: 1 - one, 1: one}, index=features.index)
            positive = 1
        predictor_path = kwargs["predictor_path"]
        predictor_path.mkdir(parents=True)
        (predictor_path / "fake.txt").write_text("fake", encoding="utf-8")

        best_model = best_hyperparameters = None
        if model.family == "ENSEMBLE":
            families = model.params["families"]
            trained_models = [f"{TOKEN[f]}_BAG_L1" for f in families] + ["WeightedEnsemble_L2"]
            best_model = "WeightedEnsemble_L2"
            best_hyperparameters = dict(model.params)
        else:
            trained_models = [TOKEN[model.family]]
            if kwargs.get("hpo") is not None:
                best_model = f"{trained_models[0]}/T1"
                best_hyperparameters = {"learning_rate": 0.05}
        return AutoGluonOutput(
            predictions=predictions,
            probabilities=probabilities,
            positive_class=positive,
            trained_models=trained_models,
            autogluon_version="1.6.fake",
            best_model=best_model,
            best_hyperparameters=best_hyperparameters,
        )


class EnsembleFinalAdapter:
    calls: list[dict[str, Any]] = []

    def fit_final(self, **kwargs: Any) -> AutoGluonOutput:
        type(self).calls.append(kwargs)
        features = kwargs["test_features"]
        model = kwargs["model"]
        best_hp = kwargs["best_hyperparameters"]
        one = features["x"].map(lambda value: 0.9 if value % 2 else 0.1)
        predictor_path = kwargs["predictor_path"]
        predictor_path.mkdir(parents=True)
        (predictor_path / "fake.txt").write_text("fake", encoding="utf-8")
        if model.family == "ENSEMBLE":
            trained = [f"{TOKEN[f]}_BAG_L1_FULL" for f in best_hp["families"]] + [
                "WeightedEnsemble_L2_FULL"
            ]
            best_model = "WeightedEnsemble_L2_FULL"
        else:
            trained = [f"{TOKEN[model.family]}_FULL"]
            best_model = trained[0]
        return AutoGluonOutput(
            predictions=features["x"].map(lambda value: value % 2),
            probabilities=pd.DataFrame({0: 1 - one, 1: one}, index=features.index),
            positive_class=1,
            trained_models=trained,
            autogluon_version="1.6.fake",
            best_model=best_model,
            best_hyperparameters=best_hp,
        )


def ensemble_project(root: Path, *, models: list[dict[str, Any]] | None = None) -> Path:
    raw = project_config(models=models or [GBM, RF, ensemble_model()])
    path = materialize(root, raw=raw)
    hpo_raw(path, top_n=5)
    return path


def run_train(path: Path) -> None:
    EnsembleAwareAdapter.calls = []
    EnsembleAwareAdapter.failures = set()
    result = train_experiment(
        build_experiment_plan(load_config(path)), adapter_factory=EnsembleAwareAdapter
    )
    assert result.is_successful


def test_train_leaderboard_includes_ensemble_as_a_candidate(tmp_path: Path) -> None:
    path = ensemble_project(tmp_path)
    run_train(path)
    manifest = json.loads((tmp_path / ".mltool/training/manifest.json").read_text())
    assert "ENSEMBLE" in {m["family"] for m in manifest["models"]}
    leaderboard = json.loads((tmp_path / ".mltool/training/leaderboard.json").read_text())
    assert {row["family"] for row in leaderboard} == {"GBM", "RF", "ENSEMBLE"}
    result = json.loads(
        (tmp_path / ".mltool/training/candidates/base__ag_ensemble/result.json").read_text()
    )
    assert result["status"] == "SUCCEEDED"
    assert result["model"]["family"] == "ENSEMBLE"
    assert "WeightedEnsemble_L2" in result["trained_models"]


def test_tune_refits_ensemble_without_hpo_and_flags_it(tmp_path: Path) -> None:
    path = ensemble_project(tmp_path)
    run_train(path)
    plan = build_experiment_plan(load_config(path))
    selection = build_tuning_selection(plan)
    assert any(c.model.family == "ENSEMBLE" for c in selection.candidates)
    EnsembleAwareAdapter.calls = []
    result = tune_experiment(plan, selection, adapter_factory=EnsembleAwareAdapter)
    assert result.is_successful
    ensemble_call = next(
        c for c in EnsembleAwareAdapter.calls if c["model"].family == "ENSEMBLE"
    )
    assert ensemble_call["hpo"] is None  # forced off despite plan.config.hpo being set
    ensemble_row = next(
        r for r in result.candidate_results if r["model"]["family"] == "ENSEMBLE"
    )
    assert ensemble_row["hpo_effective"] is False
    assert ensemble_row["hpo_warning"] == ensemble_hpo_not_applied_warning()
    assert "ENSEMBLE" in ensemble_row["hpo_warning"] or "ensemble" in ensemble_row["hpo_warning"]
    assert ensemble_row["best_hyperparameters"] == {
        "families": ["GBM", "CAT", "XGB", "RF", "XT"], "num_bag_folds": 3, "num_stack_levels": 1
    }
    # RF/XT-style messages are untouched by the new helper
    assert hpo_not_applicable_warning("RF") is not None
    assert hpo_not_applicable_warning("GBM") is None


def test_rf_hpo_call_is_unaffected_by_ensemble_support(tmp_path: Path) -> None:
    """The dispatch that forces hpo=None for ENSEMBLE must not touch RF/XT."""
    path = ensemble_project(tmp_path, models=[RF, ensemble_model()])
    run_train(path)
    plan = build_experiment_plan(load_config(path))
    selection = build_tuning_selection(plan)
    EnsembleAwareAdapter.calls = []
    tune_experiment(plan, selection, adapter_factory=EnsembleAwareAdapter)
    rf_call = next(c for c in EnsembleAwareAdapter.calls if c["model"].family == "RF")
    assert rf_call["hpo"] is not None  # unchanged: RF still receives the hpo kwargs object
    assert rf_call["hpo"].num_trials == plan.config.hpo.num_trials


def finalize_ensemble(path: Path) -> Any:
    plan = build_experiment_plan(load_config(path))
    return finalize_experiment(plan, load_finalize_input(plan), adapter_factory=EnsembleFinalAdapter)


def test_finalize_and_registry_and_mlflow_record_the_ensemble_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = ensemble_project(tmp_path, models=[ensemble_model(families=["GBM", "RF"])])
    run_train(path)
    plan = build_experiment_plan(load_config(path))
    tune_experiment(plan, build_tuning_selection(plan), adapter_factory=EnsembleAwareAdapter)

    real_read = pd.read_parquet

    def guarded(path_arg: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        assert Path(path_arg).name != "test.parquet"
        return real_read(path_arg, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded)
    EnsembleFinalAdapter.calls = []
    result = finalize_ensemble(path)
    assert result.result["model"]["family"] == "ENSEMBLE"
    assert len(EnsembleFinalAdapter.calls) == 1  # test evaluated exactly once

    selected = json.loads((tmp_path / ".mltool/tuning/selected.json").read_text())
    assert selected["best_hyperparameters"] == {
        "families": ["GBM", "RF"], "num_bag_folds": 3, "num_stack_levels": 1
    }
    final_result = json.loads((tmp_path / ".mltool/final/result.json").read_text())
    assert final_result["best_hyperparameters"] == selected["best_hyperparameters"]
    assert final_result["model"]["family"] == "ENSEMBLE"
    final_manifest = json.loads((tmp_path / ".mltool/final/manifest.json").read_text())
    assert final_manifest["selected"]["best_hyperparameters"] == selected["best_hyperparameters"]
    assert final_manifest["test_data_used"] is True

    registered = register_final(load_config(path))
    assert registered.metadata["best_hyperparameters"] == selected["best_hyperparameters"]
    assert registered.metadata["selected"]["model"]["family"] == "ENSEMBLE"

    from mltool import tracking

    run_ids, warning = tracking.log_final(load_config(path), result)
    assert warning is None and len(run_ids) == 1
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=tracking.tracking_uri(tmp_path))
    experiment = client.get_experiment_by_name(tracking.experiment_name(load_config(path)))
    run = client.get_run(run_ids[0])
    assert run.data.params["model_family"] == "ENSEMBLE"
    assert json.loads(run.data.params["hp.families"]) == ["GBM", "RF"]
    assert run.data.params["hp.num_bag_folds"] == "3"
    assert run.data.params["hp.num_stack_levels"] == "1"
    assert run.data.tags["test_data_used"] == "true"


# =========================== Fix 1: accurate "Tuned" line for ENSEMBLE =========


def test_tuned_line_wording_for_ensemble_vs_rf() -> None:
    from mltool.finalize import tuned_line

    assert tuned_line({"hpo_effective": False, "model": {"family": "ENSEMBLE"}}) == (
        "Tuned: no (HPO is not applied to ENSEMBLE candidates)"
    )
    # RF/XT wording is unchanged.
    assert tuned_line({"hpo_effective": False, "model": {"family": "RF"}}) == (
        "Tuned: no (family RF has no HPO search space)"
    )
    assert tuned_line({"hpo_effective": True, "model": {"family": "ENSEMBLE"}}) == "Tuned: yes"


def test_finalize_final_result_and_best_report_the_ensemble_wording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = ensemble_project(tmp_path, models=[ensemble_model(families=["GBM", "RF"])])
    monkeypatch.setattr(
        "mltool.cli.train_experiment",
        lambda plan: train_experiment(plan, adapter_factory=EnsembleAwareAdapter),
    )
    monkeypatch.setattr(
        "mltool.cli.tune_experiment",
        lambda plan, sel: tune_experiment(plan, sel, adapter_factory=EnsembleAwareAdapter),
    )
    monkeypatch.setattr(
        "mltool.cli.finalize_experiment",
        lambda plan, inp: finalize_experiment(plan, inp, adapter_factory=EnsembleFinalAdapter),
    )
    assert train_project(path) == 0
    assert tune_project(path) == 0
    capsys.readouterr()
    assert finalize_project(path) == 0
    expected = "Tuned: no (HPO is not applied to ENSEMBLE candidates)"
    out = capsys.readouterr().out
    assert expected in out and "has no HPO search space" not in out

    capsys.readouterr()
    assert final_result_project(path) == 0
    assert expected in capsys.readouterr().out

    assert register_project(path) == 0
    capsys.readouterr()
    assert best_project(path) == 0
    out = capsys.readouterr().out
    assert out.count(expected) == 2  # the finalized model and the registered version


def test_finalize_cli_and_register_cli_work_with_an_ensemble_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = ensemble_project(tmp_path)
    monkeypatch.setattr(
        "mltool.cli.train_experiment",
        lambda plan: train_experiment(plan, adapter_factory=EnsembleAwareAdapter),
    )
    monkeypatch.setattr(
        "mltool.cli.tune_experiment",
        lambda plan, sel: tune_experiment(plan, sel, adapter_factory=EnsembleAwareAdapter),
    )
    monkeypatch.setattr(
        "mltool.cli.finalize_experiment",
        lambda plan, inp: finalize_experiment(plan, inp, adapter_factory=EnsembleFinalAdapter),
    )
    assert train_project(path) == 0
    assert tune_project(path) == 0
    assert finalize_project(path) == 0
    assert register_project(path) == 0
    assert (tmp_path / ".mltool/registry/1/manifest.json").is_file()


# =========================== test.parquet guarantee ============================


def test_only_preparation_and_features_modules_reference_test_parquet_still_holds() -> None:
    src = Path(__file__).parents[1] / "src/mltool"
    referencing = {p.name for p in src.glob("*.py") if "test.parquet" in p.read_text()}
    assert referencing <= {"preparation.py", "feature_materialization.py"}
    for name in ("autogluon_adapter.py", "training.py", "tuning.py", "finalize.py"):
        assert name not in referencing


# =========================== Prior phases are untouched ========================


def test_phase1_through_phase7_commands_still_work_with_no_ensemble(tmp_path: Path) -> None:
    path = materialize(tmp_path, raw=project_config(models=[GBM, RF]))
    hpo_raw(path)
    RecordingAdapter.failures = set()
    assert train_experiment(
        build_experiment_plan(load_config(path)), adapter_factory=RecordingAdapter
    ).is_successful
