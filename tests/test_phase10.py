"""Phase 10: user-declared HPO search spaces (fake predictors only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from autogluon.common import space

from mltool.autogluon_adapter import AutoGluonAdapter, AutoGluonError, effective_search_space
from mltool.cli import final_result_project, register_project, tune_project
from mltool.config import ConfigError, HpoConfig, ModelConfig, TaskConfig, TrainingConfig, load_config
from mltool.experiment import build_experiment_plan
from mltool.finalize import FinalizeError, load_finalize_input, load_persisted_final
from mltool.training import load_persisted_leaderboard
from mltool.tracking import search_space_text
from mltool.tuning import (
    build_tuning_selection,
    is_tunable,
    load_persisted_tuning,
)
from test_audit_fixes import build, edit, fake_adapters  # noqa: F401  (autouse fixture)
from test_phase4 import FakePredictor, project_config, write_project
from test_phase5 import TuningAdapter
from test_phase7 import by_phase

LR = {"type": "real", "low": 0.005, "high": 0.2, "log": True}
LEAVES = {"type": "int", "low": 16, "high": 128}
EXTRA = {"type": "categorical", "values": [True, False]}
DEPTH = {"type": "int", "low": 2, "high": 8}

GBM_SS = {"name": "lightgbm", "family": "GBM", "params": {"min_data_in_leaf": 20},
          "search_space": {"learning_rate": LR, "num_leaves": LEAVES, "extra_trees": EXTRA}}
RF_SS = {"name": "forest", "family": "RF", "params": {}, "search_space": {"max_depth": DEPTH}}
RF_PLAIN = {"name": "forest", "family": "RF", "params": {}}


def models_of(tmp_path: Path, models: list[dict[str, Any]], **hpo: Any) -> Any:
    raw = project_config(models=models)
    if hpo:
        raw["hpo"] = {"top_n": 2, "num_trials": 4, "time_limit_seconds": 30, **hpo}
    return load_config(write_project(tmp_path, raw=raw))


def tuning(root: Path, cid: str) -> dict[str, Any]:
    return json.loads((root / ".mltool/tuning/candidates" / cid / "result.json").read_text())


# =========================== config ==============================================


def test_search_space_is_parsed_and_normalized(tmp_path: Path) -> None:
    model = models_of(tmp_path, [{**GBM_SS, "search_space": {
        **GBM_SS["search_space"],
        "feature_fraction": {"type": "real", "low": 0.5, "high": 1, "default": 0.9},
        "extra_trees": {**EXTRA, "default": False},
    }}]).models[0]
    assert model.params == {"min_data_in_leaf": 20}
    assert model.search_space == {
        "learning_rate": {"type": "real", "low": 0.005, "high": 0.2, "log": True},
        "num_leaves": {"type": "int", "low": 16, "high": 128},
        "extra_trees": {"type": "categorical", "values": [True, False], "default": False},
        "feature_fraction": {"type": "real", "low": 0.5, "high": 1.0, "log": False, "default": 0.9},
    }


def test_no_search_space_is_an_empty_mapping_and_hpo_searcher_defaults_to_random(
    tmp_path: Path,
) -> None:
    config = models_of(tmp_path, [RF_PLAIN], top_n=1)
    assert config.models[0].search_space == {} and config.hpo.searcher == "random"
    assert models_of(tmp_path / "g", [RF_PLAIN], searcher="grid").hpo.searcher == "grid"


@pytest.mark.parametrize("family", ["GBM", "CAT", "XGB", "RF", "XT"])
def test_search_space_is_allowed_for_every_single_family(tmp_path: Path, family: str) -> None:
    model = models_of(tmp_path, [{"name": "m", "family": family, "search_space": {"x": DEPTH}}])
    assert model.models[0].search_space == {"x": DEPTH}


def bad(spec: Any, **model: Any) -> dict[str, Any]:
    return {"name": "m", "family": "GBM", "params": {}, "search_space": {"p": spec}, **model}


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (bad({"type": "real", "low": 0.2, "high": 0.2}), "low.*less than.*high"),
        (bad({"type": "real", "low": 0.5, "high": 0.1}), "low.*less than.*high"),
        (bad({"type": "int", "low": 8, "high": 2}), "low.*less than.*high"),
        (bad({"type": "real", "low": 0, "high": 1, "log": True}), "log.*greater than 0"),
        (bad({"type": "real", "low": -1, "high": 1, "log": True}), "log.*greater than 0"),
        (bad({"type": "real", "low": 0.1, "high": 1, "log": "yes"}), "log.*true or false"),
        (bad({"type": "real", "low": "a", "high": 1}), "must be numbers"),
        (bad({"type": "real", "low": True, "high": 2}), "must be numbers"),
        (bad({"type": "int", "low": 1.5, "high": 4}), "must be integers"),
        (bad({"type": "int", "low": True, "high": 4}), "must be integers"),
        (bad({"type": "int", "low": 1, "high": 4, "log": True}), "unsupported.*log"),
        (bad({"type": "real", "low": 0.1, "high": 1, "default": 2}), "default.*within"),
        (bad({"type": "int", "low": 1, "high": 4, "default": 9}), "default.*within"),
        (bad({"type": "int", "low": 1, "high": 4, "default": 2.5}), "default.*integer"),
        (bad({"type": "categorical", "values": []}), "non-empty list"),
        (bad({"type": "categorical", "values": "ab"}), "non-empty list"),
        (bad({"type": "categorical", "values": ["a", "a"]}), "unique"),
        (bad({"type": "categorical", "values": [[1], 2]}), "JSON scalars"),
        (bad({"type": "categorical", "values": ["a"], "default": "b"}), "one of"),
        (bad({"type": "real", "low": 0.1, "high": 1, "step": 2}), "unsupported.*step"),
        (bad({"type": "uniform", "low": 0, "high": 1}), "type.*real, int, categorical"),
        (bad(5), "must be a mapping"),
        ({"name": "m", "family": "GBM", "search_space": [1]}, "mapping with non-empty"),
        (bad(DEPTH, params={"p": 3}), "both set: p"),
        ({"name": "m", "family": "GBM", "search_space": {"seed": DEPTH}}, "seed.*cannot be searched"),
        ({"name": "m", "family": "ENSEMBLE", "search_space": {"p": DEPTH}}, "not supported for ENSEMBLE"),
        ({"name": "m", "family": "GBM", "searchspace": {}}, 'unsupported "models\\[0\\]"'),
    ],
)
def test_invalid_search_spaces_are_rejected(tmp_path: Path, model: Any, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        models_of(tmp_path, [model])


def test_true_and_one_are_distinct_categorical_values(tmp_path: Path) -> None:
    spec = {"type": "categorical", "values": [True, 1, None, "1"], "default": 1}
    model = models_of(tmp_path, [bad(spec)]).models[0]
    assert model.search_space["p"]["values"] == [True, 1, None, "1"]


def test_an_unknown_searcher_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match='"hpo.searcher" must be one of: random, grid'):
        models_of(tmp_path, [RF_PLAIN], searcher="bayes")


# =========================== adapter =============================================


def fit_kwargs(tmp_path: Path, model: ModelConfig, hpo: HpoConfig | None) -> dict[str, Any]:
    FakePredictor.fit_kwargs = {}
    adapter = AutoGluonAdapter(predictor_factory=FakePredictor, version_resolver=lambda: "t")
    train = pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]})
    try:
        adapter.fit_predict(
            train_data=train, validation_features=pd.DataFrame({"x": [4, 5]}),
            task=TaskConfig("binary", "label", "yes"), model=model, primary_metric="roc_auc",
            predictor_path=tmp_path / "p", training=TrainingConfig(seed=7), hpo=hpo,
        )
    except AutoGluonError:
        pass  # FakePredictor lacks HPO outputs; the fit kwargs are already captured
    return FakePredictor.fit_kwargs


def gbm_model(tmp_path: Path) -> ModelConfig:
    return models_of(tmp_path, [GBM_SS]).models[0]


def test_space_objects_of_the_right_type_reach_the_fit_call(tmp_path: Path) -> None:
    kwargs = fit_kwargs(tmp_path, gbm_model(tmp_path / "c"), HpoConfig(2, 4, 30))
    params = kwargs["hyperparameters"]["GBM"]
    lr, leaves, extra = params["learning_rate"], params["num_leaves"], params["extra_trees"]
    assert type(lr) is space.Real and (lr.lower, lr.upper, lr.log) == (0.005, 0.2, True)
    assert type(leaves) is space.Int and (leaves.lower, leaves.upper) == (16, 128)
    assert type(extra) is space.Categorical and extra.data == [True, False]
    assert params["min_data_in_leaf"] == 20  # fixed params stay fixed
    assert params["seed"] == 7
    assert kwargs["hyperparameter_tune_kwargs"] == {
        "num_trials": 4, "scheduler": "local", "searcher": "random",
    }


def test_grid_maps_to_the_local_grid_searcher(tmp_path: Path) -> None:
    kwargs = fit_kwargs(tmp_path, gbm_model(tmp_path / "c"), HpoConfig(2, 4, 30, "grid"))
    assert kwargs["hyperparameter_tune_kwargs"]["searcher"] == "local_grid"


def test_train_never_receives_a_search_space(tmp_path: Path) -> None:
    params = fit_kwargs(tmp_path, gbm_model(tmp_path / "c"), None)["hyperparameters"]["GBM"]
    assert not any(isinstance(value, space.Space) for value in params.values())
    assert params == {"min_data_in_leaf": 20, "seed": 7}


def test_finalize_refits_fixed_hyperparameters_only(tmp_path: Path) -> None:
    class Raising(FakePredictor):
        def fit(self, **kwargs: Any) -> "Raising":
            type(self).fit_kwargs = kwargs
            raise RuntimeError("captured")

    adapter = AutoGluonAdapter(predictor_factory=Raising, version_resolver=lambda: "t")
    train = pd.DataFrame({"x": [1, 2], "label": ["no", "yes"]})
    with pytest.raises(AutoGluonError, match="captured"):
        adapter.fit_final(
            train_data=train, test_features=train[["x"]], task=TaskConfig("binary", "label", "yes"),
            model=gbm_model(tmp_path), best_hyperparameters={"learning_rate": 0.03},
            effective_seed=7, primary_metric="roc_auc", predictor_path=tmp_path / "p",
            training=TrainingConfig(seed=7),
        )
    params = Raising.fit_kwargs["hyperparameters"]["GBM"]
    assert params == {"learning_rate": 0.03, "seed": 7}


def test_a_categorical_default_is_tried_first(tmp_path: Path) -> None:
    model = models_of(tmp_path, [bad({**EXTRA, "default": False})]).models[0]
    params = fit_kwargs(tmp_path, model, HpoConfig(1, 2, 30))["hyperparameters"]["GBM"]
    assert params["p"].data == [False, True] and params["p"].default is False


def test_effective_space_merges_the_declared_space_into_the_default(tmp_path: Path) -> None:
    effective = effective_search_space(gbm_model(tmp_path), "binary")
    # min_data_in_leaf is fixed in params, so it leaves AutoGluon's default space;
    # learning_rate/num_leaves are replaced by the declared ranges; feature_fraction stays.
    assert set(effective) == {"extra_trees", "feature_fraction", "learning_rate", "num_leaves"}
    assert effective["feature_fraction"]["source"] == "autogluon_default"
    assert effective["learning_rate"] == {
        "type": "real", "low": 0.005, "high": 0.2, "log": True, "default": 0.005, "source": "user",
    }
    assert effective["num_leaves"]["high"] == 128 and effective["num_leaves"]["source"] == "user"


@pytest.mark.parametrize(
    ("family", "defaults"),
    [
        ("GBM", {"feature_fraction", "learning_rate", "min_data_in_leaf", "num_leaves"}),
        ("CAT", {"depth", "l2_leaf_reg", "learning_rate"}),
        ("XGB", {"colsample_bytree", "learning_rate", "max_depth", "min_child_weight"}),
        ("RF", set()),
        ("XT", set()),
        ("ENSEMBLE", set()),
    ],
)
def test_autogluon_default_spaces_per_family(family: str, defaults: set[str]) -> None:
    params = {"families": ["GBM"], "num_bag_folds": 2, "num_stack_levels": 0} if family == "ENSEMBLE" else {}
    effective = effective_search_space(ModelConfig("m", family, params), "binary")
    assert set(effective) == defaults
    assert all(spec["source"] == "autogluon_default" for spec in effective.values())


def test_search_space_text_matches_the_mlflow_format() -> None:
    assert search_space_text(LR) == "real[0.005,0.2,log]"
    assert search_space_text({**LEAVES, "default": 31}) == "int[16,128,default=31]"
    assert search_space_text(EXTRA) == "categorical[true,false]"


# =========================== tunability ==========================================


def test_tunability_is_family_default_or_a_declared_space(tmp_path: Path) -> None:
    rf_ss, rf, xt = models_of(tmp_path, [RF_SS, {**RF_PLAIN, "name": "rf2"},
                                         {"name": "xt", "family": "XT"}]).models
    assert is_tunable(rf_ss) and not is_tunable(rf) and not is_tunable(xt)
    assert is_tunable(ModelConfig("g", "GBM"))
    assert not is_tunable(ModelConfig("e", "ENSEMBLE", {"families": ["GBM"]}))


def test_rf_with_a_search_space_is_tuned_not_carried_over(tmp_path: Path) -> None:
    root = tmp_path
    build(root, models=[GBM_SS, RF_SS], through="train")
    assert tune_project(root / "mltool.yaml") == 0
    rf_calls = [c for c in TuningAdapter.calls if c["model"].family == "RF"]
    assert len(rf_calls) == 1 and rf_calls[0]["hpo"] is not None
    assert rf_calls[0]["model"].search_space == {"max_depth": DEPTH}
    result = tuning(root, "base__forest")
    assert result["hpo_effective"] is True and result["hpo_warning"] is None
    assert "carried_over_from_training" not in result
    assert result["search_space"] == {"max_depth": DEPTH}
    assert result["effective_search_space"] == {
        "max_depth": {"type": "int", "low": 2, "high": 8, "default": 2, "source": "user"}
    }


def test_rf_without_a_search_space_is_still_carried_over(tmp_path: Path) -> None:
    build(tmp_path, models=[GBM_SS, RF_PLAIN], through="tune")
    assert all(c["model"].family != "RF" for c in TuningAdapter.calls)
    result = tuning(tmp_path, "base__forest")
    assert result["carried_over_from_training"] is True and result["hpo_effective"] is False
    assert "declare a search_space to tune it" in result["hpo_warning"]
    assert result["search_space"] == {} and result["effective_search_space"] == {}


# =========================== records =============================================


def test_search_space_is_recorded_everywhere(tmp_path: Path) -> None:
    path = build(tmp_path, models=[GBM_SS, RF_SS], through="finalize")
    assert register_project(path) == 0
    root = tmp_path
    manifest = json.loads((root / ".mltool/tuning/manifest.json").read_text())
    assert manifest["search_spaces"] == {
        "lightgbm": load_config(path).models[0].search_space, "forest": {"max_depth": DEPTH},
    }
    assert manifest["hpo"]["searcher"] == "random"
    selected = json.loads((root / ".mltool/tuning/selected.json").read_text())
    name = selected["model"]["name"]
    declared = next(m for m in load_config(path).models if m.name == name).search_space
    assert selected["search_space"] == declared and selected["effective_search_space"]
    final = json.loads((root / ".mltool/final/result.json").read_text())
    final_manifest = json.loads((root / ".mltool/final/manifest.json").read_text())
    registry = json.loads((root / ".mltool/registry/1/metadata.json").read_text())
    for record in (final, final_manifest["selected"], registry["selected"]):
        assert record["search_space"] == declared
        assert record["effective_search_space"] == selected["effective_search_space"]

    runs = {r.data.tags["candidate_id"]: r for r in by_phase(root, "tune")}
    gbm = runs["base__lightgbm"].data.params
    assert gbm["ss.learning_rate"] == "real[0.005,0.2,log]"
    assert gbm["ss.num_leaves"] == "int[16,128]"
    assert gbm["ss.extra_trees"] == "categorical[true,false]"
    assert gbm["ss_default.feature_fraction"] == "real[0.75,1.0,default=1.0]"
    assert "ss_default.min_data_in_leaf" not in gbm  # fixed in params, so not searched
    assert gbm["searcher"] == "random"
    assert runs["base__forest"].data.params["ss.max_depth"] == "int[2,8]"
    final_run = by_phase(root, "final")[0].data.params
    assert any(key.startswith("ss.") for key in final_run)


def test_tune_report_shows_what_was_searched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = build(tmp_path, models=[GBM_SS, RF_SS], through="train")
    capsys.readouterr()
    assert tune_project(path) == 0
    out = capsys.readouterr().out
    assert "searched: declared extra_trees, learning_rate, num_leaves; AutoGluon default feature_fraction" in out
    assert "searched: declared max_depth" in out


# =========================== staleness ===========================================


def change_space(path: Path, model: str, spec: dict[str, Any] | None) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        for entry in raw["models"]:
            if entry["name"] == model:
                if spec is None:
                    entry.pop("search_space", None)
                else:
                    entry["search_space"] = spec
    edit(path, mutate)


def test_a_search_space_change_leaves_train_fresh_but_tune_and_finalize_stale(
    tmp_path: Path,
) -> None:
    path = build(tmp_path, models=[GBM_SS, RF_SS], through="tune")
    change_space(path, "forest", {"max_depth": {"type": "int", "low": 2, "high": 12}})
    config = load_config(path)
    assert load_persisted_leaderboard(config).warning is None  # train is fresh
    build_tuning_selection(build_experiment_plan(config))  # and tune may use it
    assert load_persisted_tuning(config).warning is not None
    with pytest.raises(FinalizeError, match="search_space differs.*mltool tune"):
        load_finalize_input(build_experiment_plan(config))


def test_a_searcher_change_makes_tune_and_finalize_stale(tmp_path: Path) -> None:
    path = build(tmp_path, models=[GBM_SS, RF_SS], through="tune")
    edit(path, lambda raw: raw["hpo"].__setitem__("searcher", "grid"))
    config = load_config(path)
    assert load_persisted_tuning(config).warning is not None
    with pytest.raises(FinalizeError, match="hpo configuration differs"):
        load_finalize_input(build_experiment_plan(config))


def test_the_selected_models_search_space_change_makes_the_final_model_stale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = build(tmp_path, models=[GBM_SS, RF_SS], through="finalize")
    selected = json.loads((tmp_path / ".mltool/final/manifest.json").read_text())["selected"]
    other = "forest" if selected["model"]["name"] == "lightgbm" else "lightgbm"
    change_space(path, other, None)  # another model's space: the final model is untouched
    assert load_persisted_final(load_config(path)).warning is None
    change_space(path, selected["model"]["name"], {"learning_rate": LR})
    warning = load_persisted_final(load_config(path)).warning
    assert warning == "the selected model's search space changed after this final model was created"
    capsys.readouterr()
    assert register_project(path) == 2
    assert final_result_project(path) == 0 and "search space changed" in capsys.readouterr().out


def test_artifacts_from_before_search_spaces_are_fresh_without_one(tmp_path: Path) -> None:
    path = build(tmp_path, through="finalize")  # GBM + RF, no search spaces
    for rel, drop in ((".mltool/tuning/manifest.json", "search_spaces"),):
        manifest = json.loads((tmp_path / rel).read_text())
        manifest.pop(drop)
        (tmp_path / rel).write_text(json.dumps(manifest), encoding="utf-8")
    final_path = tmp_path / ".mltool/final/manifest.json"
    final = json.loads(final_path.read_text())
    final["selected"].pop("search_space")
    final_path.write_text(json.dumps(final), encoding="utf-8")
    config = load_config(path)
    assert load_persisted_tuning(config).warning is None
    assert load_persisted_final(config).warning is None
