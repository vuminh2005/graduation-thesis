"""Phase 12: the correctness defects of the whole-system audit at 0db511f.

Fix 1: CV reproduces the split only with the prepared split settings, and
       training freshness pins the split, the source files and the CV folds.
Fix 2: the preprocessor and plugin files are hashed like SKLEARN model files.
Fix 3: the final model is stale if and only if finalize would refuse (or it was
       not built from today's inputs); status/register/final-result/best agree.
Fix 4: a registry version is self-contained, and ``mltool score`` uses it alone.

One fake adapter serves every command. Its "predictor" is a rule saved to
``predictor/rule.json``, so ``mltool score`` reloads it through the real
``AutoGluonAdapter.predict_saved`` and reproduces finalize's test predictions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
import yaml

import mltool.cross_validation as cross_validation_module
import mltool.experiment as experiment_module
from mltool.autogluon_adapter import AutoGluonAdapter, AutoGluonOutput
from mltool.cli import (
    features_project,
    final_result_project,
    finalize_project,
    main,
    plan_project,
    prepare_project,
    register_project,
    train_project,
    tune_project,
)
from mltool.config import ConfigError, load_config, source_sha256
from mltool.cross_validation import CrossValidationError, build_cv_plan
from mltool.data import load_dataset
from mltool.evaluation import evaluate_predictions
from mltool.experiment import ExperimentError, build_experiment_plan
from mltool.finalize import finalize_experiment, finalize_refusal, load_persisted_final
from mltool.registry import list_versions
from mltool.reporting import _phase_states, render_best, render_status
from mltool.scoring import ScoringError, build_features, load_scoring_record, score
from mltool.splitting import split_dataset
from mltool.state import list_runs
from mltool.training import train_experiment
from mltool.tuning import TuningError, build_tuning_selection, tune_experiment
from mltool.validation import validate_dataset

# =========================== the fake pipeline ===================================


class RuleLoader:
    """What ``TabularPredictor`` is to the real adapter: ``load`` a saved rule."""

    def __init__(self, rule: dict[str, Any]) -> None:
        self.rule = rule

    @classmethod
    def load(cls, path: str) -> "RuleLoader":
        return cls(json.loads((Path(path) / "rule.json").read_text()))

    def _scaled(self, frame: pd.DataFrame) -> pd.Series:
        rule = self.rule
        values = pd.to_numeric(frame[rule["column"]], errors="coerce").astype(float)
        spread = rule["hi"] - rule["lo"] or 1.0
        return ((values - rule["lo"]) / spread).clip(0.02, 0.98)

    def predict_proba(self, frame: pd.DataFrame, as_multiclass: bool = True) -> pd.DataFrame:
        s = self._scaled(frame)
        if self.rule["task"] == "binary":
            return pd.DataFrame({0: 1 - s, 1: s}, index=frame.index)
        raw = np.column_stack([1 - s, 0.5 * np.ones(len(s)), s])
        return pd.DataFrame(raw / raw.sum(axis=1, keepdims=True), index=frame.index,
                            columns=[0, 1, 2])

    def predict_from_proba(self, proba: pd.DataFrame) -> pd.Series:
        return proba.idxmax(axis=1)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        return self._scaled(frame) * 100.0


class RuleAdapter:
    """Fake adapter: learns a min/max rule on the fit rows' first numeric column."""

    calls: list[dict[str, Any]] = []

    def _fit(self, train: pd.DataFrame, features: pd.DataFrame, kw: dict[str, Any]) -> AutoGluonOutput:
        type(self).calls.append(kw)
        task = kw["task"]
        column = next(c for c in features.columns if pd.api.types.is_numeric_dtype(features[c]))
        values = pd.to_numeric(train[column], errors="coerce")
        rule = {"task": task.type, "column": column, "lo": float(values.min()),
                "hi": float(values.max())}
        kw["predictor_path"].mkdir(parents=True)
        (kw["predictor_path"] / "rule.json").write_text(json.dumps(rule))
        loaded = RuleLoader(rule)
        if task.type == "regression":
            predictions, probabilities, positive = loaded.predict(features), None, None
        else:
            probabilities = loaded.predict_proba(features)
            predictions = loaded.predict_from_proba(probabilities)
            positive = 1 if task.type == "binary" else None
        hpo = kw.get("hpo") is not None
        best = {"learning_rate": 0.05} if hpo else kw.get("best_hyperparameters")
        if not hpo and best is None and kw["model"].family in {"RF", "SKLEARN"}:
            best = dict(kw["model"].params)
        return AutoGluonOutput(
            predictions=predictions, probabilities=probabilities, positive_class=positive,
            trained_models=["Fake"], autogluon_version="fake",
            best_model="Fake/T1" if hpo else None, best_hyperparameters=best,
        )

    def fit_predict(self, **kw: Any) -> AutoGluonOutput:
        return self._fit(kw["train_data"], kw["validation_features"], kw)

    def fit_final(self, **kw: Any) -> AutoGluonOutput:
        return self._fit(kw["train_data"], kw["test_features"], kw)


def loader_adapter() -> AutoGluonAdapter:
    return AutoGluonAdapter(predictor_factory=RuleLoader, version_resolver=lambda: "fake")


_train, _tune, _finalize = train_experiment, tune_experiment, finalize_experiment


@pytest.fixture(autouse=True)
def fake_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    RuleAdapter.calls = []
    monkeypatch.setattr(
        "mltool.cli.train_experiment", lambda plan: _train(plan, adapter_factory=RuleAdapter)
    )
    monkeypatch.setattr(
        "mltool.cli.tune_experiment",
        lambda plan, sel: _tune(plan, sel, adapter_factory=RuleAdapter),
    )
    monkeypatch.setattr(
        "mltool.cli.finalize_experiment",
        lambda plan, inp: _finalize(plan, inp, adapter_factory=RuleAdapter),
    )
    monkeypatch.setattr(
        "mltool.scoring.AutoGluonAdapter", lambda: loader_adapter()
    )


PRE_PY = '''
import numpy as np


class Pre:
    """Learns the train median of x; fills and rescales it."""

    def fit(self, X):
        self.median_ = float(X["x"].median())
        return self

    def transform(self, X):
        out = X.copy()
        out["x"] = out["x"].fillna(self.median_) * 2.0
        return out
'''

PLUGINS_PY = '''
import pandas as pd


class Centered:
    """z minus its train mean: replaces z (a plugin input) by a new column."""

    def fit(self, X):
        self.mean_ = float(X["z"].mean())
        return self

    def transform(self, X):
        return pd.DataFrame({"z_centered": X["z"] - self.mean_}, index=X.index)


class Flag:
    def fit(self, X):
        return self

    def transform(self, X):
        return pd.DataFrame({"w_flag": (X["w"] > 1).astype(int)}, index=X.index)
'''

MODELS_PY = '''
from sklearn.linear_model import LogisticRegression


def logistic(**params):
    return LogisticRegression(max_iter=1000, **params)
'''


def dataset(task: str = "binary", rows: int = 150) -> pd.DataFrame:
    x = [(i * 37) % 101 for i in range(rows)]
    if task == "binary":
        label = [int(v + (i * 13) % 40 > 70) for i, v in enumerate(x)]
    elif task == "multiclass":
        label = [min(2, int((v + (i * 13) % 30) // 45)) for i, v in enumerate(x)]
    else:
        label = [float(v) * 1.5 + (i % 7) for i, v in enumerate(x)]
    return pd.DataFrame({
        "row_id": [1000 + i for i in range(rows)],
        "x": x,
        "z": [(i * 11) % 17 for i in range(rows)],
        "w": [i % 3 for i in range(rows)],
        "label": label,
    })


GBM = {"name": "gbm", "family": "GBM", "params": {},
       "search_space": {"num_leaves": {"type": "int", "low": 8, "high": 32}}}
RF = {"name": "rf", "family": "RF", "params": {}}
LOGIT = {"name": "logit", "family": "SKLEARN", "entrypoint": "./models.py:logistic",
         "params": {}}


def raw_config(task: str = "binary", *, cv: bool = True, sklearn: bool = True) -> dict[str, Any]:
    metrics = {"binary": ("roc_auc", ["accuracy"]), "multiclass": ("accuracy", ["f1_macro"]),
               "regression": ("rmse", ["r2"])}[task]
    evaluation: dict[str, Any] = {"primary_metric": metrics[0], "secondary_metrics": metrics[1]}
    if cv:
        evaluation["cv"] = {"folds": 3, "repeats": 1}
    task_block: dict[str, Any] = {"type": task, "target": "label"}
    if task == "binary":
        task_block["positive_class"] = 1
    return {
        "schema_version": "0.1",
        "project": {"name": "phase12"},
        "task": task_block,
        "data": {"format": "auto", "path": "./data/dataset.csv", "id_columns": ["row_id"]},
        "validation": {"enabled": True, "fail_on_error": True},
        "split": {"validation_ratio": 0.15, "test_ratio": 0.15, "stratify": "auto",
                  "random_seed": 42},
        "preprocessing": {"external": {"enabled": True, "entrypoint": "./pre.py:Pre",
                                       "params": {}}},
        "features": {
            "plugins": [
                {"name": "centered", "entrypoint": "./plugins.py:Centered", "params": {}},
                {"name": "flag", "entrypoint": "./plugins.py:Flag", "params": {}},
            ],
            "sets": [
                {"name": "base", "source_columns": ["z", "x"], "plugins": []},
                {"name": "eng", "source_columns": ["x"], "plugin_inputs": ["z", "w"],
                 "plugins": ["centered", "flag"]},
            ],
        },
        "models": [GBM, RF, *([LOGIT] if sklearn and task != "regression" else [])],
        "evaluation": evaluation,
        "training": {"seed": 42},
        "hpo": {"top_n": 6, "num_trials": 4, "time_limit_seconds": 30},
    }


def write(root: Path, raw: dict[str, Any], frame: pd.DataFrame | None = None) -> Path:
    (root / "data").mkdir(parents=True, exist_ok=True)
    (dataset(raw["task"]["type"]) if frame is None else frame).to_csv(
        root / "data/dataset.csv", index=False
    )
    (root / "pre.py").write_text(PRE_PY, encoding="utf-8")
    (root / "plugins.py").write_text(PLUGINS_PY, encoding="utf-8")
    (root / "models.py").write_text(MODELS_PY, encoding="utf-8")
    path = root / "mltool.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


STEPS: dict[str, Callable[[Path], int]] = {
    "prepare": prepare_project, "features": features_project, "plan": plan_project,
    "train": train_project, "tune": tune_project, "finalize": finalize_project,
    "register": register_project,
}


def pipeline(root: Path, raw: dict[str, Any] | None = None, through: str = "register") -> Path:
    path = write(root, raw or raw_config())
    names = list(STEPS)
    for name in names[: names.index(through) + 1]:
        assert STEPS[name](path) == 0, name
    RuleAdapter.calls = []
    return path


def edit(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    raw = yaml.safe_load(path.read_text())
    mutate(raw)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def replace_text(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new))


def states(path: Path) -> dict[str, str]:
    return {phase: state for phase, (state, _) in _phase_states(load_config(path)).items()}


def selected_set(root: Path) -> str:
    return json.loads((root / ".mltool/final/manifest.json").read_text())["selected"]["feature_set"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def test_the_fake_pipeline_selects_a_plugin_feature_set(tmp_path: Path) -> None:
    """The scenarios below rely on the selected set having plugins and plugin_inputs."""
    path = pipeline(tmp_path)
    assert selected_set(tmp_path) == "eng"
    assert states(path)["register"] == "fresh"


# =========================== Fix 1: the CV split-seed leak =======================


def test_the_audit_leak_sequence_is_refused_before_any_fit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """prepare(42) -> features -> seed 43 without prepare -> train: refused, no fit."""
    path = pipeline(tmp_path, through="features")
    edit(path, lambda raw: raw["split"].__setitem__("random_seed", 43))
    capsys.readouterr()
    assert train_project(path) == 2
    out = capsys.readouterr().out
    assert "split configuration differs" in out and 'run "mltool prepare"' in out
    assert RuleAdapter.calls == []
    assert not (tmp_path / ".mltool/training").exists()
    assert plan_project(path) == 2  # plan refuses as well: features rest on a stale prepare
    with pytest.raises(CrossValidationError, match="split configuration differs"):
        build_cv_plan(load_config(path))
    # the revert leaves nothing behind that could pass for a fresh training run
    edit(path, lambda raw: raw["split"].__setitem__("random_seed", 42))
    assert states(path)["train"] == "missing"


def test_a_training_run_from_the_leak_is_stale_after_the_revert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training artifacts produced by the old code at seed 43 (reproduced here by
    disabling the new split check) are stale once the seed is back at 42."""
    path = pipeline(tmp_path, through="features")
    prepared_test = pd.read_parquet(tmp_path / ".mltool/prepared/test.parquet")
    edit(path, lambda raw: raw["split"].__setitem__("random_seed", 43))
    with monkeypatch.context() as old_code:
        noop = lambda config: json.loads(  # noqa: E731
            (tmp_path / ".mltool/prepared/manifest.json").read_text()
        )
        old_code.setattr(experiment_module, "validate_prepared_manifest", noop)
        old_code.setattr(cross_validation_module, "validate_prepared_manifest", noop)
        leaked = build_cv_plan(load_config(path))
        assert train_project(path) == 0
    assert RuleAdapter.calls  # it did fit
    # the development set it scored on contains prepared test rows: the leak
    assert set(leaked.development["row_id"]) & set(prepared_test["row_id"])
    manifest = load_json(tmp_path / ".mltool/training/manifest.json")
    assert manifest["cv"]["fold_fingerprint"] == leaked.fingerprint

    edit(path, lambda raw: raw["split"].__setitem__("random_seed", 42))
    RuleAdapter.calls = []
    report = states(path)
    assert report["prepare"] == "fresh" and report["features"] == "fresh"
    assert report["train"] == "stale"
    with pytest.raises(TuningError, match="split settings differ"):
        build_tuning_selection(build_experiment_plan(load_config(path)))
    assert tune_project(path) == 2 and RuleAdapter.calls == []


def test_the_fold_fingerprint_alone_decides_for_a_legacy_training_manifest(tmp_path: Path) -> None:
    """A manifest from before the split was recorded is judged by its CV folds."""
    path = pipeline(tmp_path, through="train")
    manifest_path = tmp_path / ".mltool/training/manifest.json"
    manifest = load_json(manifest_path)
    del manifest["split"]
    manifest_path.write_text(json.dumps(manifest))
    assert states(path)["train"] == "fresh"  # the folds prove the split
    manifest["cv"]["fold_fingerprint"] = "0" * 64  # as trained on another split
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(TuningError, match="cross-validation folds differ"):
        build_tuning_selection(build_experiment_plan(load_config(path)))


def test_holdout_training_also_refuses_a_stale_prepare(tmp_path: Path) -> None:
    path = pipeline(tmp_path, raw_config(cv=False), through="features")
    edit(path, lambda raw: raw["split"].__setitem__("test_ratio", 0.2))
    assert train_project(path) == 2 and RuleAdapter.calls == []


def test_training_records_what_its_scores_depend_on(tmp_path: Path) -> None:
    path = pipeline(tmp_path, through="train")
    manifest = load_json(tmp_path / ".mltool/training/manifest.json")
    assert manifest["split"] == {"random_seed": 42, "validation_ratio": 0.15,
                                 "test_ratio": 0.15, "stratified": True}
    assert manifest["preprocessor_source_sha256"] == source_sha256(tmp_path / "pre.py")
    plan = build_experiment_plan(load_config(path))
    assert {e["name"]: e["recipe_fingerprint"] for e in manifest["feature_sets"]} == {
        a.name: a.recipe_fingerprint for a in plan.feature_sets
    }


# =========================== Fix 2: source hashing ===============================


def test_prepare_and_features_record_the_source_hashes(tmp_path: Path) -> None:
    pipeline(tmp_path, through="features")
    prepared = load_json(tmp_path / ".mltool/prepared/manifest.json")
    assert prepared["preprocessing"]["source_sha256"] == source_sha256(tmp_path / "pre.py")
    eng = load_json(tmp_path / ".mltool/features/eng/manifest.json")
    assert [p["source_sha256"] for p in eng["plugins"]] == [source_sha256(tmp_path / "plugins.py")] * 2


def test_a_preprocessor_edit_makes_prepare_and_everything_after_it_stale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = pipeline(tmp_path)
    replace_text(tmp_path / "pre.py", "* 2.0", "* 3.0")
    assert set(states(path).values()) - {"n/a"} == {"stale"}
    capsys.readouterr()
    for name in ("features", "plan", "train", "tune", "finalize"):
        args = {"force": True} if name == "finalize" else {}
        assert STEPS[name](path, **args) == 2, name
        assert "preprocessor source file changed" in capsys.readouterr().out, name
    assert register_project(path) == 2
    assert RuleAdapter.calls == [] and list_versions(tmp_path) == [1]
    # re-running the fix path makes everything fresh again
    for name in ("prepare", "features", "train", "tune"):
        assert STEPS[name](path) == 0, name
    assert finalize_project(path, force=True) == 0 and register_project(path) == 0
    assert set(states(path).values()) - {"n/a"} == {"fresh"}


def test_a_plugin_edit_makes_features_and_everything_after_stale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = pipeline(tmp_path)
    replace_text(tmp_path / "plugins.py", "- self.mean_", "- self.mean_ + 1.0")
    report = states(path)
    assert report["prepare"] == "fresh"
    assert {report[p] for p in ("features", "train", "tune", "finalize", "register")} == {"stale"}
    capsys.readouterr()
    assert plan_project(path) == 2
    assert 'plugin "centered" source file changed' in capsys.readouterr().out
    assert finalize_project(path, force=True) == 2 and register_project(path) == 2
    assert RuleAdapter.calls == []
    warning = load_persisted_final(load_config(path)).warning
    assert warning is not None and "recipe changed" in warning  # the selected set's


def test_old_manifests_without_hashes_are_stale_only_when_code_ran(tmp_path: Path) -> None:
    path = pipeline(tmp_path, through="train")
    prepared_path = tmp_path / ".mltool/prepared/manifest.json"
    prepared = load_json(prepared_path)
    del prepared["preprocessing"]["source_sha256"]
    prepared_path.write_text(json.dumps(prepared))
    assert states(path)["prepare"] == "stale"  # a preprocessor ran: unprovable
    edit(path, lambda raw: raw["preprocessing"].__setitem__(
        "external", {"enabled": False, "entrypoint": None, "params": {}}))
    prepared["preprocessing"] = {"enabled": False}
    prepared_path.write_text(json.dumps(prepared))
    assert states(path)["prepare"] == "fresh"  # none ran: nothing to prove

    other = pipeline(tmp_path / "plugins", through="train")
    set_path = tmp_path / "plugins/.mltool/features/eng/manifest.json"
    manifest = load_json(set_path)
    for plugin in manifest["plugins"]:
        del plugin["source_sha256"]
    set_path.write_text(json.dumps(manifest))
    with pytest.raises(ExperimentError, match="predates plugin source hashing"):
        build_experiment_plan(load_config(other))


def test_old_training_manifests_are_stale_only_when_unprovable(tmp_path: Path) -> None:
    path = pipeline(tmp_path, through="train")
    manifest_path = tmp_path / ".mltool/training/manifest.json"
    original = load_json(manifest_path)

    def with_(mutate: Callable[[dict[str, Any]], None]) -> str:
        manifest = json.loads(json.dumps(original))
        mutate(manifest)
        manifest_path.write_text(json.dumps(manifest))
        return states(path)["train"]

    assert with_(lambda m: m.pop("preprocessor_source_sha256")) == "stale"  # a preprocessor ran
    def drop_recipes(m: dict[str, Any]) -> None:
        for entry in m["feature_sets"]:
            del entry["recipe_fingerprint"]
    assert with_(drop_recipes) == "stale"  # "eng" has plugins
    def drop_base_recipe(m: dict[str, Any]) -> None:
        del m["feature_sets"][0]["recipe_fingerprint"]  # "base": no plugins
    assert with_(drop_base_recipe) == "fresh"


# =========================== Fix 3: one staleness rule ===========================


def _set_selected(key: str, value: Any) -> Callable[[Path], None]:
    def mutate(path: Path) -> None:
        name = selected_set(path.parent)

        def change(raw: dict[str, Any]) -> None:
            for spec in raw["features"]["sets"]:
                if spec["name"] == name:
                    spec[key] = value

        edit(path, change)
    return mutate


def _config(fn: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
    return lambda path: edit(path, fn)


def _append_row(path: Path) -> None:
    csv = path.parent / "data/dataset.csv"
    frame = pd.read_csv(csv)
    frame.loc[len(frame)] = frame.iloc[0]
    frame.to_csv(csv, index=False)


def _reprepare_with_new_seed(path: Path) -> None:
    edit(path, lambda raw: raw["split"].__setitem__("random_seed", 43))
    assert prepare_project(path) == 0


MATRIX: dict[str, Callable[[Path], None]] = {
    # the audit's eight mutations
    "preprocessor_source": lambda p: replace_text(p.parent / "pre.py", "* 2.0", "* 3.0"),
    "plugin_source": lambda p: replace_text(p.parent / "plugins.py", "(X[\"w\"] > 1)", "(X[\"w\"] > 0)"),
    "sklearn_model_source": lambda p: replace_text(p.parent / "models.py", "1000", "2000"),
    "search_space_gbm": _config(lambda r: r["models"][0]["search_space"]["num_leaves"].__setitem__("high", 64)),
    "cv_folds": _config(lambda r: r["evaluation"].__setitem__("cv", {"folds": 4, "repeats": 1})),
    "plugin_inputs": _set_selected("plugin_inputs", ["z", "w", "row_id"]),
    "training_seed": _config(lambda r: r["training"].__setitem__("seed", 43)),
    "split_seed_no_prepare": _config(lambda r: r["split"].__setitem__("random_seed", 43)),
    # inconsistencies the audit listed, and more upstream changes
    "hpo_num_trials": _config(lambda r: r["hpo"].__setitem__("num_trials", 5)),
    "hpo_searcher": _config(lambda r: r["hpo"].__setitem__("searcher", "grid")),
    "primary_metric": _config(lambda r: r["evaluation"].update(
        primary_metric="accuracy", secondary_metrics=["roc_auc"])),
    "preprocessing_params": _config(lambda r: r["preprocessing"]["external"].__setitem__("params", {"k": 1})),
    "other_feature_set_recipe": _config(lambda r: r["features"]["sets"][0].__setitem__("source_columns", ["x"])),
    "dataset_changed": _append_row,
    "reprepared_split": _reprepare_with_new_seed,
}


@pytest.mark.parametrize("change", list(MATRIX))
def test_final_staleness_equals_finalize_refusal(tmp_path: Path, change: str) -> None:
    path = pipeline(tmp_path)
    MATRIX[change](path)
    stale = load_persisted_final(load_config(path)).warning is not None
    refuses = finalize_project(path, force=True) == 2
    assert RuleAdapter.calls == [], "refused before any fit"
    assert stale and refuses
    report = states(path)
    assert report["finalize"] == "stale" and report["register"] == "stale"
    assert register_project(path) == 2 and list_versions(tmp_path) == [1]


def test_unchanged_inputs_keep_the_final_fresh_and_finalize_runs(tmp_path: Path) -> None:
    path = pipeline(tmp_path)
    assert load_persisted_final(load_config(path)).warning is None
    assert finalize_refusal(load_config(path)) is None
    assert finalize_project(path, force=True) == 0 and len(RuleAdapter.calls) == 1


def test_a_final_model_from_an_older_selection_is_stale_although_finalize_would_run(
    tmp_path: Path,
) -> None:
    """The other half of the rule: finalize would happily run, but this final
    model was built from a tuning selection that has since been replaced."""
    path = pipeline(tmp_path)
    selected_path = tmp_path / ".mltool/tuning/selected.json"
    selected = load_json(selected_path)
    selected["tuned_validation_score"] -= 0.01  # a re-run of tune with another outcome
    selected_path.write_text(json.dumps(selected))
    config = load_config(path)
    assert finalize_refusal(config) is None
    assert "tuning selection differs" in load_persisted_final(config).warning


def test_final_result_best_and_register_share_the_rule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = pipeline(tmp_path)
    MATRIX["hpo_num_trials"](path)
    capsys.readouterr()
    assert final_result_project(path) == 0
    assert "finalize would refuse" in capsys.readouterr().out
    assert "finalize would refuse" in render_best(load_config(path))
    assert register_project(path) == 2
    assert register_project(path, force=True) == 0
    metadata = load_json(tmp_path / ".mltool/registry/2/metadata.json")
    assert metadata["forced"] is True and "finalize would refuse" in metadata["warning"]


# =========================== status ==============================================


def test_status_propagates_a_stale_upstream_phase(tmp_path: Path) -> None:
    path = pipeline(tmp_path)
    assert set(states(path).values()) - {"n/a"} == {"fresh"}
    edit(path, lambda raw: raw["split"].__setitem__("random_seed", 43))
    report = _phase_states(load_config(path))
    assert [report[p][0] for p in ("prepare", "features", "train", "tune", "finalize", "register")] \
        == ["stale"] * 6
    assert "split configuration differs" in report["prepare"][1]
    text = render_status(load_config(path))
    assert "register: " in text


def test_status_register_is_fresh_only_for_the_current_fresh_final(tmp_path: Path) -> None:
    path = pipeline(tmp_path)
    assert _phase_states(load_config(path))["register"] == ("fresh", "1 version(s), latest 1")
    assert finalize_project(path, force=True) == 0
    # a new final model, not yet registered (edited, as the fake refit is byte-identical)
    manifest_path = tmp_path / ".mltool/final/manifest.json"
    manifest = load_json(manifest_path)
    manifest["note"] = "refit"
    manifest_path.write_text(json.dumps(manifest))
    state, note = _phase_states(load_config(path))["register"]
    assert state == "stale" and "not the current final model" in note
    assert register_project(path) == 0
    assert _phase_states(load_config(path))["register"][0] == "fresh"
    shutil.rmtree(tmp_path / ".mltool/final")
    assert _phase_states(load_config(path))["register"][0] == "stale"


# =========================== Fix 4: registry and score ===========================


def test_the_final_result_and_manifest_record_how_to_score_raw_rows(tmp_path: Path) -> None:
    pipeline(tmp_path)
    final = tmp_path / ".mltool/final"
    result, manifest = load_json(final / "result.json"), load_json(final / "manifest.json")
    assert result["scoring"] == manifest["scoring"]
    record = manifest["scoring"]
    assert record["target"] == "label" and record["task"] == "binary"
    assert record["positive_class"] == 1 and record["id_columns"] == ["row_id"]
    assert record["raw_feature_columns"] == ["row_id", "x", "z", "w"]
    assert record["preprocessor"] == "preprocessor.pkl"
    assert record["feature_set"] == {
        "name": "eng", "source_columns": ["x"], "plugin_inputs": ["z", "w"],
        "plugins": [
            {"name": "centered", "artifact": "feature_plugins/centered.pkl",
             "generated_columns": ["z_centered"]},
            {"name": "flag", "artifact": "feature_plugins/flag.pkl",
             "generated_columns": ["w_flag"]},
        ],
        "final_feature_columns": ["x", "z_centered", "w_flag"],
    }


def _path_fields(value: Any, key: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        return [item for k, v in value.items() for item in _path_fields(v, k)]
    if isinstance(value, list):
        return [item for v in value for item in _path_fields(v, key)]
    return [(key, value)] if isinstance(value, str) else []


# Absolute by design: where the raw data and the user's code live, and MLflow's store.
ABSOLUTE_ALLOWED = {"resolved_entrypoint", "path", "tracking_uri"}


def test_artifact_paths_are_relative_to_their_manifest(tmp_path: Path) -> None:
    pipeline(tmp_path)
    root = str(tmp_path)
    checked = 0
    for manifest in (tmp_path / ".mltool").rglob("*.json"):
        if "mlflow" in manifest.parts:
            continue
        data = load_json(manifest)
        for key, value in _path_fields(data):
            if root in value:
                assert key in ABSOLUTE_ALLOWED and "registry" not in value, (manifest, key, value)
                if key == "path":  # only the raw dataset's location
                    assert manifest.name == "manifest.json" and value.endswith("dataset.csv")
    for manifest in (tmp_path / ".mltool").rglob("manifest.json"):
        artifacts = load_json(manifest).get("artifacts")
        if isinstance(artifacts, dict):
            for rel in [artifacts["predictor"], artifacts["preprocessor"],
                        *artifacts["feature_plugins"].values()]:
                assert (manifest.parent / rel).exists(), (manifest, rel)
                checked += 1
    assert checked == 8  # .mltool/final and registry/1: predictor, preprocessor, 2 plugins
    training = load_json(tmp_path / ".mltool/training/manifest.json")
    assert (tmp_path / ".mltool/training" / training["feature_manifest"]).is_file()
    tuning = load_json(tmp_path / ".mltool/tuning/manifest.json")
    assert (tmp_path / ".mltool/tuning" / tuning["training_manifest"]).is_file()
    selected = load_json(tmp_path / ".mltool/tuning/selected.json")
    if selected["predictor_path"]:
        assert (tmp_path / ".mltool/tuning" / selected["predictor_path"]).is_dir()


def raw_test_rows(path: Path) -> pd.DataFrame:
    config = load_config(path)
    frame = load_dataset(config.data).frame
    return split_dataset(frame, target="label", task_type=config.task.type, config=config.split).test


def _recorded_vs_scored(tmp_path: Path, task: str) -> None:
    path = pipeline(tmp_path / "project", raw_config(task))
    project = path.parent
    test_rows = raw_test_rows(path)
    test_rows.to_csv(tmp_path / "raw_test.csv", index=False)
    recorded = load_json(project / ".mltool/final/result.json")["metrics"]
    config = load_config(path)

    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(project / ".mltool/registry", elsewhere / "registry")
    project.rename(tmp_path / "moved")  # mltool.yaml, .mltool/final and the user's files are gone
    os.chdir(elsewhere)
    output = elsewhere / "predictions.csv"
    assert main(["score", "--input", str(tmp_path / "raw_test.csv"), "--output", str(output),
                 "--registry", "registry"]) == 0
    scored = pd.read_csv(output)
    assert scored["row_id"].tolist() == test_rows["row_id"].tolist()
    probabilities = None
    if task != "regression":
        classes = [0, 1] if task == "binary" else [0, 1, 2]
        assert [c for c in scored.columns if c.startswith("proba_")] == [f"proba_{c}" for c in classes]
        probabilities = pd.DataFrame({c: scored[f"proba_{c}"] for c in classes})
    metrics = evaluate_predictions(
        task_type=task, evaluation=config.evaluation, y_true=test_rows["label"].reset_index(drop=True),
        predictions=scored["prediction"], probabilities=probabilities,
        positive_class=1 if task == "binary" else None,
    )
    assert metrics == pytest.approx(recorded, abs=1e-12)


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_score_from_a_registry_copy_reproduces_the_recorded_test_metrics(
    tmp_path: Path, task: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # restored after the test's own chdir
    _recorded_vs_scored(tmp_path, task)


FRESH_PROCESS = r'''
import json, sys
from pathlib import Path
import pandas as pd
from mltool.scoring import build_features, load_scoring_record
version = Path(sys.argv[1])
rows = pd.read_csv(sys.argv[2])
features = build_features(load_scoring_record(1, version), version, rows)
features.to_csv(sys.argv[3], index=False)
print(json.dumps(sorted(m for m in sys.modules if m.startswith("_mltool_"))))
'''


def test_the_fitted_transforms_replay_in_a_fresh_process_without_the_users_files(
    tmp_path: Path,
) -> None:
    """No AutoGluon here: the preprocessor and plugins alone, loaded by value."""
    path = pipeline(tmp_path / "project")
    rows = raw_test_rows(path)
    rows.to_csv(tmp_path / "rows.csv", index=False)
    version = tmp_path / "version"
    shutil.copytree(tmp_path / "project/.mltool/registry/1", version)
    expected = build_features(load_scoring_record(1, version), version, rows)
    (tmp_path / "project").rename(tmp_path / "gone")
    done = subprocess.run(
        [sys.executable, "-c", FRESH_PROCESS, str(version), str(tmp_path / "rows.csv"),
         str(tmp_path / "features.csv")],
        capture_output=True, text=True, cwd="/", timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []  # no user module imported
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "features.csv"), expected.reset_index(drop=True),
        check_dtype=False,
    )


def test_score_errors_are_clear_and_write_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = pipeline(tmp_path)
    registry = tmp_path / ".mltool/registry"
    rows = raw_test_rows(path)
    rows.drop(columns=["z"]).to_csv(tmp_path / "missing.csv", index=False)
    with pytest.raises(ScoringError, match="missing required column.*z"):
        score(registry=registry, input_path=tmp_path / "missing.csv",
              output_path=tmp_path / "out.csv")
    assert not (tmp_path / "out.csv").exists()
    rows.to_csv(tmp_path / "rows.csv", index=False)
    with pytest.raises(ScoringError, match="version 7 does not exist"):
        score(registry=registry, input_path=tmp_path / "rows.csv",
              output_path=tmp_path / "out.csv", version=7)
    with pytest.raises(ScoringError, match="output must be a .csv or .parquet"):
        score(registry=registry, input_path=tmp_path / "rows.csv",
              output_path=tmp_path / "out.txt")
    legacy = tmp_path / "legacy/1"
    shutil.copytree(registry / "1", legacy)
    manifest = load_json(legacy / "manifest.json")
    del manifest["scoring"]
    (legacy / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ScoringError, match="predates self-contained scoring"):
        score(registry=tmp_path / "legacy", input_path=tmp_path / "rows.csv",
              output_path=tmp_path / "out.csv")
    capsys.readouterr()
    assert main(["score", "--input", str(tmp_path / "missing.csv"), "--output",
                 str(tmp_path / "out.csv"), "--registry", str(registry)]) == 2
    assert "missing required column" in capsys.readouterr().out


def test_score_picks_the_latest_version_by_default_and_is_not_logged(tmp_path: Path) -> None:
    path = pipeline(tmp_path)
    assert finalize_project(path, force=True) == 0 and register_project(path) == 0
    assert list_versions(tmp_path) == [1, 2]
    runs_before = len(list_runs(tmp_path))
    raw_test_rows(path).to_parquet(tmp_path / "rows.parquet", index=False)
    result = score(registry=tmp_path / ".mltool/registry", input_path=tmp_path / "rows.parquet",
                   output_path=tmp_path / "out.parquet")
    assert result.version == 2
    assert pd.read_parquet(tmp_path / "out.parquet").columns[0] == "row_id"
    assert len(list_runs(tmp_path)) == runs_before


# =========================== data.id_columns =====================================


def test_id_columns_are_validated(tmp_path: Path) -> None:
    raw = raw_config()
    path = write(tmp_path, raw)
    assert load_config(path).data.id_columns == ["row_id"]
    for bad, message in ((["label"], "may not contain the target"), (["a", "a"], "unique"),
                         ("row_id", "list of non-empty strings")):
        edit(path, lambda r: r["data"].__setitem__("id_columns", bad))
        with pytest.raises(ConfigError, match=message):
            load_config(path)
    edit(path, lambda r: r["data"].__setitem__("id_columns", ["nope"]))
    config = load_config(path)
    report = validate_dataset(config, load_dataset(config.data))
    assert 'id column "nope" (data.id_columns) was not found' in report.errors
    edit(path, lambda r: r["data"].pop("id_columns"))
    assert load_config(path).data.id_columns == []
