"""Phase 11: sklearn-compatible custom models (fake predictors; no AutoGluon fit)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from mltool.autogluon_adapter import (
    AutoGluonAdapter,
    AutoGluonError,
    AutoGluonOutput,
    _check_trained_models,
    effective_model_seed,
)
from mltool.cli import plan_project, register_project, tune_project
from mltool.config import ConfigError, HpoConfig, TaskConfig, TrainingConfig, load_config, model_record
from mltool.custom_models import (
    AutoInput,
    CustomModelError,
    align_probabilities,
    load_entrypoint,
    seed_decision,
    validate,
)
from mltool.experiment import build_experiment_plan
from mltool.finalize import FinalizeError, finalize_experiment, load_finalize_input, load_persisted_final
from mltool.training import load_persisted_leaderboard, train_experiment
from mltool.tuning import TuningError, build_tuning_selection, is_tunable, tune_experiment
from test_phase4 import FakePredictor, materialize, project_config, write_project
from test_phase7 import by_phase

MODELS_PY = '''
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class Centroid:
    """A user-defined estimator: nearest class mean, no random_state."""

    def __init__(self, shrink=0.0):
        self.shrink = shrink

    def get_params(self, deep=True):
        return {"shrink": self.shrink}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.means_ = np.stack([X[y == c].mean(axis=0) for c in self.classes_]) * (1 - self.shrink)
        return self

    def predict_proba(self, X):
        distance = ((np.asarray(X, dtype=float)[:, None, :] - self.means_[None]) ** 2).sum(-1)
        weight = np.exp(-distance)
        return weight / weight.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def logistic(**params):
    return LogisticRegression(**params)


def svc(**params):
    return SVC(**params)


def broken(**params):
    raise RuntimeError("factory exploded")


NOT_CALLABLE = 3
'''

LOGISTIC = {"name": "logit", "family": "SKLEARN", "entrypoint": "./models.py:logistic",
            "params": {"C": 0.5}}
CENTROID = {"name": "centroid", "family": "SKLEARN", "entrypoint": "./models.py:Centroid"}
SVC_TUNED = {"name": "svc", "family": "SKLEARN", "entrypoint": "./models.py:svc",
             "params": {"probability": True},
             "search_space": {"C": {"type": "real", "low": 0.01, "high": 10, "log": True}}}


def project(root: Path, models: list[dict[str, Any]], **raw_extra: Any) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "models.py").write_text(MODELS_PY, encoding="utf-8")
    raw = project_config(models=models, training={"seed": 7})
    raw.update(raw_extra)
    return write_project(root, raw=raw)


def model_of(tmp_path: Path, spec: dict[str, Any]) -> Any:
    return load_config(project(tmp_path, [spec])).models[0]


# =========================== config ==============================================


def test_sklearn_model_is_parsed_with_its_entrypoint(tmp_path: Path) -> None:
    model = model_of(tmp_path, {**LOGISTIC, "input": "raw"})
    assert model.family == "SKLEARN" and model.entrypoint == "./models.py:logistic"
    assert model.source_path == (tmp_path / "models.py").resolve() and model.input == "raw"
    assert model_of(tmp_path / "d", LOGISTIC).input == "auto"


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"name": "m", "family": "SKLEARN"}, "entrypoint.*required"),
        ({**LOGISTIC, "entrypoint": "./models.py"}, "<python-file>:<class-or-function>"),
        ({**LOGISTIC, "entrypoint": "./models.py:not-a-name"}, "<python-file>:<class-or-function>"),
        ({**LOGISTIC, "entrypoint": "./missing.py:logistic"}, "file not found"),
        ({**LOGISTIC, "input": "scaled"}, "input.*auto, raw"),
        ({**LOGISTIC, "params": {"mltool_seed": 1}}, "reserved"),
        ({**LOGISTIC, "search_space": {"random_state": {"type": "int", "low": 1, "high": 3}}},
         "random_state.*cannot be searched"),
        ({**LOGISTIC, "search_space": {"tol": {"type": "real", "low": 2, "high": 1}}}, "less than"),
        ({"name": "g", "family": "GBM", "entrypoint": "./models.py:logistic"}, "unsupported.*entrypoint"),
        ({"name": "g", "family": "GBM", "input": "raw"}, "unsupported.*input"),
        ({"name": "e", "family": "ENSEMBLE", "params": {"families": ["GBM", "SKLEARN"]}},
         "cannot be ENSEMBLE members"),
    ],
)
def test_invalid_sklearn_configs_are_rejected(tmp_path: Path, spec: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        model_of(tmp_path, spec)


# =========================== contract validation =================================


def test_an_svc_without_probability_is_rejected_for_classification(tmp_path: Path) -> None:
    model = model_of(tmp_path, {**SVC_TUNED, "params": {}})
    with pytest.raises(CustomModelError, match=r"no predict_proba\(\).*probability: true"):
        validate(model, "binary")
    validate(model, "regression")  # regression needs only fit/predict


@pytest.mark.parametrize(
    ("entrypoint", "message"),
    [
        ("./models.py:missing", '"missing" was not found'),
        ("./models.py:NOT_CALLABLE", "not a class or function"),
        ("./models.py:broken", "factory exploded"),
    ],
)
def test_bad_entrypoints_are_clear_errors(tmp_path: Path, entrypoint: str, message: str) -> None:
    model = model_of(tmp_path, {**LOGISTIC, "entrypoint": entrypoint, "params": {}})
    with pytest.raises(CustomModelError, match=message):
        validate(model, "binary")


def test_plan_reports_a_contract_violation_as_a_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "models.py").write_text(MODELS_PY, encoding="utf-8")
    path = materialize(tmp_path, raw=project_config(models=[{**SVC_TUNED, "params": {}}]))
    capsys.readouterr()
    assert plan_project(path) == 2
    assert "predict_proba" in capsys.readouterr().out


def test_the_search_ranges_first_values_are_used_to_build_the_probe(tmp_path: Path) -> None:
    validate(model_of(tmp_path, SVC_TUNED), "binary")  # SVC(probability=True, C=0.01)


# =========================== input: auto =========================================


def test_input_auto_learns_only_from_the_fit_rows() -> None:
    fit = pd.DataFrame({
        "age": [10.0, 20.0, np.nan, 30.0],
        "city": pd.Categorical(["a", "a", None, "b"]),
    })
    other = pd.DataFrame({"age": [1000.0, np.nan], "city": pd.Categorical(["zzz", None])})
    auto = AutoInput().fit(fit)
    assert auto.numeric_ == ["age"] and auto.categorical_ == ["city"]
    assert auto.medians_["age"] == 20.0  # from the fit rows, not 1000
    assert auto.categories_["city"] == ["a", "b"] and auto.modes_["city"] == "a"
    out = auto.transform(other)
    assert out.shape == (2, 3)  # age + one-hot(a, b)
    assert out[0, 1:].tolist() == [0.0, 0.0]  # unseen category -> all zeros
    assert out[1, 1:].tolist() == [1.0, 0.0]  # missing -> the fit rows' most frequent
    filled = np.array([10.0, 20.0, 20.0, 30.0])
    assert out[1, 0] == pytest.approx((20.0 - filled.mean()) / filled.std())  # NaN -> fit median


def test_input_auto_statistics_move_with_the_fit_slice() -> None:
    frame = pd.DataFrame({"x": np.arange(100.0)})
    low, high = AutoInput().fit(frame.iloc[:10]), AutoInput().fit(frame.iloc[90:])
    assert low.means_["x"] == 4.5 and high.means_["x"] == 94.5
    assert not np.allclose(low.transform(frame), high.transform(frame))


# =========================== probabilities =======================================


def test_probabilities_are_aligned_to_the_encoded_classes() -> None:
    proba = np.array([[0.2, 0.8], [0.6, 0.4]])
    out = align_probabilities(proba, classes=[0, 2], n_classes=3)  # class 1 absent from the fit rows
    assert out.tolist() == [[0.2, 0.0, 0.8], [0.6, 0.0, 0.4]]
    assert align_probabilities(np.array([0.3]), None, 2).tolist() == [[0.7, 0.3]]
    with pytest.raises(ValueError, match="outside"):
        align_probabilities(proba, classes=[0, 5], n_classes=3)


# =========================== seeds ===============================================


def test_seed_rules(tmp_path: Path) -> None:
    training = TrainingConfig(seed=7)
    logistic = model_of(tmp_path / "a", LOGISTIC)
    assert seed_decision(logistic, training) == (7, "random_state set from training.seed")
    fixed = model_of(tmp_path / "b", {**LOGISTIC, "params": {"random_state": 3}})
    assert seed_decision(fixed, training) == (3, "random_state fixed in params")
    centroid = model_of(tmp_path / "c", CENTROID)
    seed, note = seed_decision(centroid, training)
    assert seed is None and "no random_state" in note
    assert seed_decision(logistic, TrainingConfig(seed=None)) == (None, "training.seed is null")
    assert effective_model_seed(logistic, training) == 7


# =========================== adapter =============================================


def fit_kwargs(tmp_path: Path, model: Any, hpo: HpoConfig | None, seed: int | None = 7) -> dict[str, Any]:
    FakePredictor.fit_kwargs = {}
    adapter = AutoGluonAdapter(predictor_factory=FakePredictor, version_resolver=lambda: "t")
    train = pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]})
    try:
        adapter.fit_predict(
            train_data=train, validation_features=pd.DataFrame({"x": [4, 5]}),
            task=TaskConfig("binary", "label", "yes"), model=model, primary_metric="roc_auc",
            predictor_path=tmp_path / "p", training=TrainingConfig(seed=seed), hpo=hpo,
        )
    except AutoGluonError:
        pass  # FakePredictor's model names are not the wrapper's; the kwargs are captured
    return FakePredictor.fit_kwargs


def test_the_wrapper_class_and_private_keys_reach_autogluon(tmp_path: Path) -> None:
    from autogluon.common import space
    from mltool.autogluon_sklearn import MLToolSklearnModel

    model = model_of(tmp_path, SVC_TUNED)
    (cls, params), = fit_kwargs(tmp_path, model, None)["hyperparameters"].items()
    assert cls is MLToolSklearnModel
    assert params == {
        "probability": True,
        "mltool_entrypoint": f"{(tmp_path / 'models.py').resolve()}:svc",
        "mltool_input": "auto",
        "mltool_random_state": 7,  # applied with set_params, not passed to the factory
    }
    assert "C" not in params  # no Space objects without a search
    (_, tuned), = fit_kwargs(tmp_path, model, HpoConfig(2, 4, 30))["hyperparameters"].items()
    assert isinstance(tuned["C"], space.Real) and tuned["C"].log is True


def test_no_seed_key_without_random_state_or_when_the_user_fixed_it(tmp_path: Path) -> None:
    (_, centroid), = fit_kwargs(tmp_path, model_of(tmp_path / "a", CENTROID), None)["hyperparameters"].items()
    assert "mltool_random_state" not in centroid
    fixed = model_of(tmp_path / "b", {**LOGISTIC, "params": {"random_state": 3}})
    (_, params), = fit_kwargs(tmp_path, fixed, None)["hyperparameters"].items()
    assert params["random_state"] == 3 and "mltool_random_state" not in params


def test_finalize_passes_the_fixed_best_hyperparameters(tmp_path: Path) -> None:
    class Raising(FakePredictor):
        def fit(self, **kwargs: Any) -> "Raising":
            type(self).fit_kwargs = kwargs
            raise RuntimeError("captured")

    adapter = AutoGluonAdapter(predictor_factory=Raising, version_resolver=lambda: "t")
    train = pd.DataFrame({"x": [1, 2], "label": ["no", "yes"]})
    with pytest.raises(AutoGluonError, match="captured"):
        adapter.fit_final(
            train_data=train, test_features=train[["x"]], task=TaskConfig("binary", "label", "yes"),
            model=model_of(tmp_path, SVC_TUNED), best_hyperparameters={"probability": True, "C": 2.5},
            effective_seed=7, primary_metric="roc_auc", predictor_path=tmp_path / "p",
            training=TrainingConfig(seed=7),
        )
    (_, params), = Raising.fit_kwargs["hyperparameters"].items()
    assert params["C"] == 2.5 and params["mltool_random_state"] == 7
    assert Raising.fit_kwargs["refit_full"] is True


def test_the_trained_model_guard_accepts_exactly_the_wrapper() -> None:
    _check_trained_models(["MLToolSklearn", "MLToolSklearn/T3", "MLToolSklearn_FULL"], "SKLEARN")
    for bad in (["LightGBM"], ["MLToolSklearn_2"], ["MLToolSklearnX"], ["MLToolSklearn", "RandomForest"]):
        with pytest.raises(AutoGluonError, match="outside family SKLEARN"):
            _check_trained_models(bad, "SKLEARN")


def test_best_hyperparameters_drop_the_private_keys() -> None:
    from mltool.autogluon_adapter import _best_hyperparameters

    class P:
        def info(self) -> dict[str, Any]:
            return {"model_info": {"MLToolSklearn/T2": {"hyperparameters": {
                "C": 3.0, "mltool_entrypoint": "/x.py:f", "mltool_input": "auto"}}}}

    assert _best_hyperparameters(P(), "MLToolSklearn/T2") == {"C": 3.0}


# =========================== persistence =========================================


def test_a_saved_wrapper_loads_in_a_fresh_process_without_the_users_file(tmp_path: Path) -> None:
    """The mechanism the registry relies on, via AutoGluon's own plain pickle."""
    from mltool.autogluon_sklearn import MLToolSklearnModel

    source = tmp_path / "user" / "models.py"
    source.parent.mkdir()
    source.write_text(MODELS_PY, encoding="utf-8")
    Centroid = load_entrypoint(f"{source}:Centroid")
    X = np.array([[0.0, 0.0], [0.1, 0.2], [3.0, 3.0], [3.1, 2.9]])
    estimator = Centroid().fit(X, np.array([0, 0, 1, 1]))
    expected = estimator.predict_proba(X)

    with pytest.raises(pickle.PicklingError):  # why by-value storage is needed at all
        pickle.dumps(estimator)

    wrapper = MLToolSklearnModel(path=str(tmp_path / "w"), name="MLToolSklearn",
                                 problem_type="binary", eval_metric="roc_auc")
    wrapper.model = estimator
    saved = tmp_path / "wrapper.pkl"
    saved.write_bytes(pickle.dumps(wrapper, protocol=4))  # as save_pkl.save does
    assert wrapper.model is estimator  # saving does not disturb the live object
    shutil.rmtree(source.parent)  # the user's file (and any __pycache__) is gone
    assert not source.exists()

    script = (
        "import pickle, sys, numpy as np\n"
        f"w = pickle.loads(open({str(saved)!r}, 'rb').read())\n"
        "assert not any('models' in (getattr(m, '__file__', '') or '') and 'user' in (getattr(m, '__file__', '') or '') for m in list(sys.modules.values()))\n"
        f"X = np.array({X.tolist()!r})\n"
        "print(np.asarray(w.model.predict_proba(X)).tolist())\n"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert np.allclose(json.loads(done.stdout.strip()), expected)


# =========================== pipeline: carry-over, staleness, records ============


class SklearnFake:
    """A generic fake adapter that trains nothing; SKLEARN-shaped model names."""

    calls: list[dict[str, Any]] = []

    def _out(self, frame: pd.DataFrame, kw: dict[str, Any], hpo: bool) -> AutoGluonOutput:
        type(self).calls.append(kw)
        p = kw["predictor_path"]
        p.mkdir(parents=True)
        (p / "fake.txt").write_text("fake")
        s = pd.Series(np.linspace(0.1, 0.9, len(frame)), index=frame.index)
        best = {**kw["model"].params, "C": 1.0} if hpo else kw.get("best_hyperparameters")
        return AutoGluonOutput(
            predictions=(s > 0.5).astype(int), probabilities=pd.DataFrame({0: 1 - s, 1: s}),
            positive_class=1, trained_models=["MLToolSklearn/T1"] if hpo else ["MLToolSklearn"],
            autogluon_version="fake", best_model="MLToolSklearn/T1" if hpo else None,
            best_hyperparameters=best,
        )

    def fit_predict(self, **kw: Any) -> AutoGluonOutput:
        return self._out(kw["validation_features"], kw, kw.get("hpo") is not None)

    def fit_final(self, **kw: Any) -> AutoGluonOutput:
        return self._out(kw["test_features"], kw, False)


def pipeline(tmp_path: Path, models: list[dict[str, Any]], through: str = "finalize") -> Path:
    (tmp_path / "models.py").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "models.py").write_text(MODELS_PY, encoding="utf-8")
    raw = project_config(models=models, training={"seed": 7})
    raw["hpo"] = {"top_n": len(models), "num_trials": 4, "time_limit_seconds": 30}
    path = materialize(tmp_path, raw=raw)
    SklearnFake.calls = []
    plan = build_experiment_plan(load_config(path))
    train_experiment(plan, adapter_factory=SklearnFake)
    if through == "train":
        return path
    plan = build_experiment_plan(load_config(path))
    tune_experiment(plan, build_tuning_selection(plan), adapter_factory=SklearnFake)
    if through == "tune":
        return path
    plan = build_experiment_plan(load_config(path))
    finalize_experiment(plan, load_finalize_input(plan), adapter_factory=SklearnFake)
    return path


def result(root: Path, phase: str, cid: str) -> dict[str, Any]:
    return json.loads((root / f".mltool/{phase}/candidates/{cid}/result.json").read_text())


def test_untuned_sklearn_is_carried_over_and_a_declared_space_is_tuned(tmp_path: Path) -> None:
    pipeline(tmp_path, [LOGISTIC, SVC_TUNED], through="train")
    config = load_config(tmp_path / "mltool.yaml")
    assert not is_tunable(config.models[0]) and is_tunable(config.models[1])
    SklearnFake.calls = []
    plan = build_experiment_plan(config)
    tune_experiment(plan, build_tuning_selection(plan), adapter_factory=SklearnFake)
    tuned_models = {call["model"].name for call in SklearnFake.calls if call.get("hpo") is not None}
    assert tuned_models == {"svc"}
    assert result(tmp_path, "tuning", "base__logit")["carried_over_from_training"] is True
    svc = result(tmp_path, "tuning", "base__svc")
    assert svc["hpo_effective"] is True and "carried_over_from_training" not in svc
    # CV/holdout re-scoring and finalize keep the entrypoint after HPO fixes the params
    assert all(call["model"].entrypoint == "./models.py:svc"
               for call in SklearnFake.calls if call["model"].name == "svc")


def test_editing_the_model_file_makes_train_tune_and_finalize_stale(tmp_path: Path) -> None:
    path = pipeline(tmp_path, [LOGISTIC])
    config = load_config(path)
    assert load_persisted_leaderboard(config).warning is None
    assert load_persisted_final(config).warning is None
    with (tmp_path / "models.py").open("a", encoding="utf-8") as stream:
        stream.write("\n# edited\n")
    config = load_config(path)
    assert load_persisted_leaderboard(config).warning is not None
    with pytest.raises(TuningError, match='"models" differs'):
        build_tuning_selection(build_experiment_plan(config))
    with pytest.raises(FinalizeError, match="stale"):
        load_finalize_input(build_experiment_plan(config))
    assert load_persisted_final(config).warning is not None


def test_changing_the_input_mode_makes_train_stale(tmp_path: Path) -> None:
    path = pipeline(tmp_path, [LOGISTIC], through="train")
    raw = yaml.safe_load(path.read_text())
    raw["models"][0]["input"] = "raw"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    assert load_persisted_leaderboard(load_config(path)).warning is not None


def test_the_sklearn_model_is_recorded_in_every_artifact(tmp_path: Path) -> None:
    path = pipeline(tmp_path, [LOGISTIC])
    assert register_project(path) == 0
    config = load_config(path)
    expected = model_record(config.models[0])
    assert expected["entrypoint"] == "./models.py:logistic" and expected["input"] == "auto"
    assert len(expected["source_sha256"]) == 64
    root = tmp_path / ".mltool"
    training = result(tmp_path, "training", "base__logit")
    tuning = result(tmp_path, "tuning", "base__logit")
    selected = json.loads((root / "tuning/selected.json").read_text())
    final = json.loads((root / "final/result.json").read_text())
    final_manifest = json.loads((root / "final/manifest.json").read_text())
    registry = json.loads((root / "registry/1/metadata.json").read_text())
    for record in (training["model"], tuning["model"], selected["model"], final["model"],
                   final_manifest["selected"]["model"], registry["selected"]["model"]):
        assert record == expected
    for manifest in ("training/manifest.json", "tuning/manifest.json", "final/manifest.json"):
        assert json.loads((root / manifest).read_text())["models"] == [expected]
    for record in (training, tuning, selected, final, registry):
        assert record["seed_note"] == "random_state set from training.seed"
    assert training["effective_seed"] == final["effective_seed"] == 7


def test_mlflow_records_the_sklearn_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mltool.cli as cli

    monkeypatch.setattr(cli, "train_experiment", lambda plan: train_experiment(plan, adapter_factory=SklearnFake))
    monkeypatch.setattr(cli, "tune_experiment", lambda plan, sel: tune_experiment(plan, sel, adapter_factory=SklearnFake))
    (tmp_path / "models.py").write_text(MODELS_PY, encoding="utf-8")
    raw = project_config(models=[LOGISTIC], training={"seed": 7})
    raw["hpo"] = {"top_n": 1, "num_trials": 4, "time_limit_seconds": 30}
    path = materialize(tmp_path, raw=raw)
    assert plan_project(path) == 0 and cli.train_project(path) == 0 and tune_project(path) == 0
    run = by_phase(tmp_path, "train")[0].data.params
    assert run["model_entrypoint"] == "./models.py:logistic" and run["model_input"] == "auto"
    assert len(run["model_source_sha256"]) == 64 and run["seed_note"].startswith("random_state set")
    assert by_phase(tmp_path, "tune")[0].data.params["model_entrypoint"] == "./models.py:logistic"


def test_other_families_keep_their_three_key_record(tmp_path: Path) -> None:
    config = load_config(project(tmp_path, [{"name": "g", "family": "GBM", "params": {"x": 1}}]))
    assert model_record(config.models[0]) == {"name": "g", "family": "GBM", "params": {"x": 1}}


def wrapper(tmp_path: Path, **params: Any) -> Any:
    from mltool.autogluon_sklearn import MLToolSklearnModel

    model = MLToolSklearnModel(path=str(tmp_path / "w"), name="MLToolSklearn",
                               problem_type="binary", eval_metric="roc_auc")
    model.params = dict(params)  # AutoGluon fills this at fit time; set it directly here
    return model


def test_the_wrapper_is_refit_on_all_rows_by_refit_full(tmp_path: Path) -> None:
    # With False, refit_full would duplicate the model instead of retraining it
    # (autogluon/core/models/abstract/_tags.py).
    assert wrapper(tmp_path)._get_tags()["can_refit_full"] is True


def test_the_wrapper_learns_input_auto_only_on_the_fit_call(tmp_path: Path) -> None:
    model = wrapper(tmp_path, mltool_input="auto")
    fit_rows = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    later = pd.DataFrame({"x": [100.0, 200.0]})
    model._preprocess(fit_rows, is_train=True)
    before = model._mltool_auto.means_["x"]
    out = model._preprocess(later)  # prediction-time call: transform only
    assert model._mltool_auto.means_["x"] == before == 2.0
    assert out.tolist() == (((later.to_numpy() - 2.0) / fit_rows["x"].std(ddof=0))).tolist()


def test_input_raw_passes_the_frame_through(tmp_path: Path) -> None:
    model = wrapper(tmp_path, mltool_input="raw")
    frame = pd.DataFrame({"x": [1.0, np.nan], "c": pd.Categorical(["a", "b"])})
    assert model._preprocess(frame, is_train=True) is frame
