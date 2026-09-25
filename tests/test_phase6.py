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
    final_result_project,
    finalize_project,
    prepare_project,
    features_project,
)
from mltool.config import ModelConfig, TaskConfig, TrainingConfig, load_config
from mltool.experiment import ExperimentError, build_experiment_plan
from mltool.finalize import (
    FinalizeError,
    finalize_experiment,
    load_finalize_input,
    load_persisted_final,
)
from mltool.training import TrainingError, train_experiment
from mltool.tuning import build_tuning_selection, tune_experiment
from test_phase4 import RecordingAdapter, materialize, project_config, write_project
from test_phase5 import HpoFakePredictor, TuningAdapter, hpo_raw

MODELS = [
    {"name": "lightgbm", "family": "GBM", "params": {}},
    {"name": "forest", "family": "RF", "params": {}},
]


class FinalFakePredictor(HpoFakePredictor):
    """HpoFakePredictor plus AutoGluon's predict_from_proba."""

    def predict_from_proba(self, proba: pd.DataFrame) -> pd.Series:
        return pd.Series(["no", "yes"], index=proba.index)


class FinalAdapter:
    """Fake adapter: records the single final fit and predicts from the x column."""

    calls: list[dict[str, Any]] = []

    def fit_final(self, **kwargs: Any) -> AutoGluonOutput:
        type(self).calls.append(kwargs)
        features = kwargs["test_features"]
        one = features["x"].map(lambda value: 0.9 if value % 2 else 0.1)
        predictor_path = kwargs["predictor_path"]
        predictor_path.mkdir(parents=True)
        (predictor_path / "fake.txt").write_text("fake", encoding="utf-8")
        return AutoGluonOutput(
            predictions=features["x"].map(lambda value: value % 2),
            probabilities=pd.DataFrame({0: 1 - one, 1: one}, index=features.index),
            positive_class=1,
            trained_models=["LightGBM", "LightGBM_FULL"],
            autogluon_version="1.6.fake",
            best_model="LightGBM_FULL",
            best_hyperparameters=kwargs["best_hyperparameters"],
        )


def write_plugins(root: Path) -> dict[str, Path]:
    plugins = root / "plugins"
    plugins.mkdir(parents=True)
    audits = {name: root / f"{name}.audit" for name in ("pre", "plug_a", "plug_b")}
    (plugins / "audit.py").write_text(
        """
import pandas as pd

def log(path, rows):
    with open(path, "a") as stream:
        stream.write(f"{rows}\\n")

class Pre:
    def __init__(self, audit_file):
        self.audit_file = audit_file
    def fit(self, X):
        log(self.audit_file, len(X)); self.mean = float(X["x"].mean())
    def transform(self, X):
        return X.copy()

class Plug:
    def __init__(self, audit_file, column):
        self.audit_file = audit_file; self.column = column
    def fit(self, X):
        log(self.audit_file, len(X)); self.mean = float(X["x"].mean())
    def transform(self, X):
        return pd.DataFrame({self.column: X["x"] - self.mean}, index=X.index)
""",
        encoding="utf-8",
    )
    return audits


def audit_rows(path: Path) -> list[int]:
    return [int(line) for line in path.read_text().split()] if path.exists() else []


def final_project(
    root: Path, *, seed: int | None = 7, plugins: bool = False, hpo: bool = True
) -> tuple[Path, dict[str, Path]]:
    audits: dict[str, Path] = {}
    features = None
    if plugins:
        audits = write_plugins(root)
        features = {
            "plugins": [
                {"name": "a", "entrypoint": "./plugins/audit.py:Plug",
                 "params": {"audit_file": str(audits["plug_a"]), "column": "xa"}},
                {"name": "b", "entrypoint": "./plugins/audit.py:Plug",
                 "params": {"audit_file": str(audits["plug_b"]), "column": "xb"}},
            ],
            "sets": [
                {"name": "with_a", "source_columns": ["*"], "plugins": ["a"]},
                {"name": "with_b", "source_columns": ["*"], "plugins": ["b"]},
            ],
        }
    raw = project_config(models=MODELS, training={"seed": seed}, features=features)
    if plugins:
        raw["preprocessing"]["external"] = {
            "enabled": True,
            "entrypoint": "./plugins/audit.py:Pre",
            "params": {"audit_file": str(audits["pre"])},
        }
    path = materialize(root, raw=raw)
    if hpo:
        hpo_raw(path, top_n=2)
    RecordingAdapter.failures = set()
    plan = build_experiment_plan(load_config(path))
    train_experiment(plan, adapter_factory=RecordingAdapter)
    TuningAdapter.failures = set()
    plan = build_experiment_plan(load_config(path))
    tune_experiment(plan, build_tuning_selection(plan), adapter_factory=TuningAdapter)
    return path, audits


def run_finalize(path: Path, adapter: Any = FinalAdapter):
    plan = build_experiment_plan(load_config(path))
    return finalize_experiment(plan, load_finalize_input(plan), adapter_factory=adapter)


# --- happy path & artifacts ---------------------------------------------------


def test_finalize_refits_on_train_plus_validation_and_persists(tmp_path: Path) -> None:
    path, _ = final_project(tmp_path)
    FinalAdapter.calls = []
    prepared = json.loads((tmp_path / ".mltool/prepared/manifest.json").read_text())["split"]
    result = run_finalize(path)

    assert len(FinalAdapter.calls) == 1  # exactly one fit
    call = FinalAdapter.calls[0]
    assert len(call["train_data"]) == prepared["train_rows"] + prepared["validation_rows"]
    assert len(call["test_features"]) == prepared["test_rows"]
    assert "label" in call["train_data"] and "label" not in call["test_features"]
    assert set(call["train_data"].index).isdisjoint(call["test_features"].index)
    assert ".final-staging-" in str(call["predictor_path"])

    selected = json.loads((tmp_path / ".mltool/tuning/selected.json").read_text())
    assert call["best_hyperparameters"] == selected["best_hyperparameters"]
    assert call["effective_seed"] == selected["effective_seed"] == 7
    assert call["model"].family == selected["model"]["family"]

    final = tmp_path / ".mltool/final"
    assert (final / "predictor/fake.txt").is_file()
    assert not list((tmp_path / ".mltool").glob(".final-staging-*"))
    stored = json.loads((final / "result.json").read_text())
    assert stored["metrics"]["roc_auc"] == 1.0
    assert stored["candidate_id"] == selected["candidate_id"]
    assert stored["best_hyperparameters"] == selected["best_hyperparameters"]
    assert stored["effective_seed"] == 7 and stored["seed"] == 7
    assert stored["rows"] == {"train_validation": 85, "test": 15}
    assert stored["refit"]["fit_split"] == "train+validation"
    assert stored["predictor_path"] == "predictor"  # relative to .mltool/final
    assert result.result["metrics"] == stored["metrics"]


def test_only_final_manifest_records_test_data_used(tmp_path: Path) -> None:
    path, _ = final_project(tmp_path)
    mltool = tmp_path / ".mltool"
    assert json.loads((mltool / "training/manifest.json").read_text())["test_data_used"] is False
    assert json.loads((mltool / "tuning/manifest.json").read_text())["test_data_used"] is False
    assert json.loads((mltool / "tuning/selected.json").read_text())["test_data_used"] is False
    run_finalize(path)
    manifest = json.loads((mltool / "final/manifest.json").read_text())
    assert manifest["test_data_used"] is True and manifest["test_evaluations"] == 1
    assert manifest["selected"]["best_hyperparameters"]
    # the earlier phases' manifests are untouched by finalize
    assert json.loads((mltool / "training/manifest.json").read_text())["test_data_used"] is False


# --- split reproduction --------------------------------------------------------


def test_raw_split_reproduction_matches_recorded_rows_and_errors_otherwise(
    tmp_path: Path,
) -> None:
    path, _ = final_project(tmp_path)
    FinalAdapter.calls = []
    run_finalize(path)  # counts agree (85 / 15)

    manifest_path = tmp_path / ".mltool/prepared/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["split"]["test_rows"] += 1
    manifest_path.write_text(json.dumps(manifest))
    FinalAdapter.calls = []
    with pytest.raises(FinalizeError, match="reproduced raw split does not match"):
        run_finalize(path)
    assert FinalAdapter.calls == []  # nothing was fit or evaluated

    manifest["split"]["test_rows"] -= 1
    manifest["split"]["train_rows"] -= 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FinalizeError, match="train\\+validation 85 vs 84"):
        run_finalize(path)


def test_split_config_change_after_prepare_is_stale(tmp_path: Path) -> None:
    path, _ = final_project(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["split"]["random_seed"] = 1
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    # Since Phase 12 the plan already refuses: features built on a stale prepare are stale.
    with pytest.raises(ExperimentError, match='split configuration differs.*"mltool prepare"'):
        run_finalize(path)


# --- refit semantics -------------------------------------------------------------


def test_preprocessor_and_selected_plugins_are_refit_on_combined_rows_only(
    tmp_path: Path,
) -> None:
    path, audits = final_project(tmp_path, plugins=True)
    # prepare fit the preprocessor on train (70); features fit each plugin on train (70).
    assert audit_rows(audits["pre"]) == [70]
    assert audit_rows(audits["plug_a"]) == [70] and audit_rows(audits["plug_b"]) == [70]
    selected = json.loads((tmp_path / ".mltool/tuning/selected.json").read_text())
    chosen, other = ("plug_a", "plug_b") if selected["feature_set"] == "with_a" else ("plug_b", "plug_a")

    FinalAdapter.calls = []
    run_finalize(path)
    assert audit_rows(audits["pre"]) == [70, 85]  # refit once, on train+validation
    assert audit_rows(audits[chosen]) == [70, 85]
    assert audit_rows(audits[other]) == [70]  # non-selected FeatureSet never refit
    call = FinalAdapter.calls[0]
    expected = "xa" if chosen == "plug_a" else "xb"
    assert expected in call["train_data"] and expected in call["test_features"]
    assert len(call["train_data"]) == 85 and len(call["test_features"]) == 15
    stored = json.loads((tmp_path / ".mltool/final/result.json").read_text())
    assert stored["refit"]["preprocessor"]["enabled"] is True
    assert stored["refit"]["preprocessor"]["fit_rows"] == 85
    assert [p["fit_rows"] for p in stored["refit"]["feature_plugins"]] == [85]


def test_refit_uses_train_validation_statistics_not_train_only(tmp_path: Path) -> None:
    path, _ = final_project(tmp_path, plugins=True)
    FinalAdapter.calls = []
    run_finalize(path)
    call = FinalAdapter.calls[0]
    column = "xa" if "xa" in call["train_data"] else "xb"
    # the plugin centres x by the fit-set mean: the fit frame must average to ~0 exactly
    assert call["train_data"][column].mean() == pytest.approx(0.0, abs=1e-9)


# --- real adapter: fixed hyperparameters, no HPO ------------------------------------


def adapter_final(tmp_path: Path, task: TaskConfig, family: str, hp: dict[str, Any],
                  seed: int | None, training: TrainingConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    FinalFakePredictor.names = ["LightGBM", "LightGBM_FULL"]
    FinalFakePredictor.model_best = "LightGBM_FULL"
    adapter = AutoGluonAdapter(predictor_factory=FinalFakePredictor, version_resolver=lambda: "t")
    output = adapter.fit_final(
        train_data=pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]}),
        test_features=pd.DataFrame({"x": [4, 5]}),
        task=task,
        model=ModelConfig("m", family, {"ignored": 1}),
        best_hyperparameters=hp,
        effective_seed=seed,
        primary_metric="roc_auc",
        predictor_path=tmp_path / "p",
        training=training,
    )
    return FinalFakePredictor.fit_kwargs, {"output": output, "init": FinalFakePredictor.init_kwargs}


def test_final_fit_uses_fixed_best_hyperparameters_and_no_hpo(tmp_path: Path) -> None:
    fit, extra = adapter_final(
        tmp_path, TaskConfig("binary", "label", "yes"), "GBM",
        {"learning_rate": 0.05, "num_leaves": 9, "seed": 1}, 99, TrainingConfig(30, 7),
    )
    assert fit["hyperparameters"] == {
        "GBM": {"learning_rate": 0.05, "num_leaves": 9, "seed": 99}  # effective_seed wins
    }
    assert fit["hyperparameter_tune_kwargs"] is None
    assert "tuning_data" not in fit and "ignored" not in fit["hyperparameters"]["GBM"]
    assert fit["num_bag_folds"] == 0 and fit["num_stack_levels"] == 0
    assert fit["dynamic_stacking"] is False and fit["fit_weighted_ensemble"] is False
    assert fit["fit_full_last_level_weighted_ensemble"] is False
    assert fit["full_weighted_ensemble_additionally"] is False
    assert fit["num_gpus"] == 0 and fit["fit_strategy"] == "sequential"
    assert fit["refit_full"] is True and fit["set_best_to_refit_full"] is True
    assert fit["time_limit"] == 30
    assert extra["init"]["learner_kwargs"] == {"random_state": 7}
    assert extra["output"].best_model == "LightGBM_FULL"


def test_final_inference_is_a_single_call_on_the_test_features(tmp_path: Path) -> None:
    calls: list[str] = []

    class Counting(FinalFakePredictor):
        def predict(self, frame: pd.DataFrame) -> pd.Series:
            calls.append("predict"); return super().predict(frame)

        def predict_proba(self, frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
            calls.append("predict_proba"); return super().predict_proba(frame, **kwargs)

        def predict_from_proba(self, proba: pd.DataFrame) -> pd.Series:
            calls.append("predict_from_proba")
            return pd.Series(["no", "yes"], index=proba.index)

    Counting.names = ["LightGBM"]
    adapter = AutoGluonAdapter(predictor_factory=Counting, version_resolver=lambda: "t")
    adapter.fit_final(
        train_data=pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]}),
        test_features=pd.DataFrame({"x": [4, 5]}),
        task=TaskConfig("binary", "label", "yes"), model=ModelConfig("m", "GBM", {}),
        best_hyperparameters={}, effective_seed=None, primary_metric="roc_auc",
        predictor_path=tmp_path / "p", training=TrainingConfig(),
    )
    assert calls == ["predict_proba", "predict_from_proba"]  # model inference happens once
    assert Counting.fit_kwargs["hyperparameters"] == {"GBM": {}}
    assert "learner_kwargs" not in Counting.init_kwargs and "time_limit" not in Counting.fit_kwargs


def test_final_fit_rejects_ensemble_or_foreign_models(tmp_path: Path) -> None:
    class Ensemble(FinalFakePredictor):
        names = ["LightGBM", "WeightedEnsemble_L2"]

    adapter = AutoGluonAdapter(predictor_factory=Ensemble, version_resolver=lambda: "t")
    with pytest.raises(AutoGluonError, match="weighted ensemble"):
        adapter.fit_final(
            train_data=pd.DataFrame({"x": [1], "label": ["no"]}),
            test_features=pd.DataFrame({"x": [4]}),
            task=TaskConfig("binary", "label", "yes"), model=ModelConfig("m", "GBM", {}),
            best_hyperparameters={}, effective_seed=None, primary_metric="roc_auc",
            predictor_path=tmp_path / "p", training=TrainingConfig(),
        )


# --- test data is touched exactly once ------------------------------------------------


def test_test_parquet_is_never_read_and_test_is_evaluated_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = final_project(tmp_path)
    plan = build_experiment_plan(load_config(path))
    finalize_input = load_finalize_input(plan)
    real_read = pd.read_parquet
    opened: list[str] = []

    def guarded(path_arg: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        opened.append(Path(path_arg).name)
        assert Path(path_arg).name != "test.parquet"
        return real_read(path_arg, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded)
    import mltool.finalize as finalize_module

    real_evaluate = finalize_module.evaluate_predictions
    evaluations: list[int] = []
    monkeypatch.setattr(
        finalize_module,
        "evaluate_predictions",
        lambda **kw: evaluations.append(len(kw["y_true"])) or real_evaluate(**kw),
    )
    FinalAdapter.calls = []
    finalize_experiment(plan, finalize_input, adapter_factory=FinalAdapter)
    assert "test.parquet" not in opened
    assert len(FinalAdapter.calls) == 1 and evaluations == [15]


def test_only_preparation_and_features_modules_reference_test_parquet() -> None:
    src = Path(__file__).parents[1] / "src/mltool"
    referencing = {p.name for p in src.glob("*.py") if "test.parquet" in p.read_text()}
    assert referencing <= {"preparation.py", "feature_materialization.py"}
    for name in ("finalize.py", "experiment.py", "training.py", "tuning.py",
                 "autogluon_adapter.py", "evaluation.py"):
        assert name not in referencing


def test_failed_final_fit_persists_nothing_and_keeps_previous_result(tmp_path: Path) -> None:
    path, _ = final_project(tmp_path)
    run_finalize(path)
    before = (tmp_path / ".mltool/final/result.json").read_bytes()

    class Exploding:
        def fit_final(self, **kwargs: Any) -> AutoGluonOutput:
            raise AutoGluonError("boom")

    with pytest.raises(FinalizeError, match="final refit failed: boom"):
        run_finalize(path, Exploding)
    assert (tmp_path / ".mltool/final/result.json").read_bytes() == before
    assert not list((tmp_path / ".mltool").glob(".final-staging-*"))


# --- refusals ---------------------------------------------------------------------


def test_finalize_refuses_when_tuning_missing_or_training_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = materialize(tmp_path / "notune", raw=project_config(models=MODELS))
    hpo_raw(path)
    train_experiment(build_experiment_plan(load_config(path)), adapter_factory=RecordingAdapter)
    capsys.readouterr()
    assert finalize_project(path) == 2
    assert 'run "mltool tune" first' in capsys.readouterr().out
    assert not (tmp_path / "notune/.mltool/final").exists()

    path = materialize(tmp_path / "nothing", raw=project_config(models=MODELS))
    hpo_raw(path)
    capsys.readouterr()
    assert finalize_project(path) == 2
    assert 'run "mltool tune" first' in capsys.readouterr().out

    path, _ = final_project(tmp_path / "notrain")
    import shutil

    shutil.rmtree(tmp_path / "notrain/.mltool/training")
    capsys.readouterr()
    assert finalize_project(path) == 2
    assert 'run "mltool train" first' in capsys.readouterr().out
    assert not (tmp_path / "notrain/.mltool/final").exists()


def test_finalize_refuses_stale_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "mltool.cli.finalize_experiment", lambda *a, **k: pytest.fail("must not finalize")
    )

    def edit(root: Path, fn) -> Path:
        path, _ = final_project(root)
        raw = yaml.safe_load(path.read_text())
        fn(raw)
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        return path

    cases = {
        "models": lambda r: r["models"][0].__setitem__("params", {"num_boost_round": 4}),
        "hpo": lambda r: r["hpo"].__setitem__("num_trials", 9),
        "seed": lambda r: r["training"].__setitem__("seed", 8),
        "metrics": lambda r: r.__setitem__(
            "evaluation", {"primary_metric": "accuracy", "secondary_metrics": ["f1"]}
        ),
    }
    for name, fn in cases.items():
        path = edit(tmp_path / name, fn)
        assert finalize_project(path) == 2, name
        assert "stale" in capsys.readouterr().out, name

    # training re-run after tuning
    path, _ = final_project(tmp_path / "retrain")
    manifest_path = tmp_path / "retrain/.mltool/training/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["note"] = "rerun"
    manifest_path.write_text(json.dumps(manifest))
    assert finalize_project(path) == 2
    assert "training was re-run after tuning" in capsys.readouterr().out

    # feature artifacts re-materialized after tuning
    path, _ = final_project(tmp_path / "features")
    frame = pd.read_parquet(tmp_path / "features/.mltool/features/base/train.parquet")
    frame.iloc[0, 0] += 1
    frame.to_parquet(tmp_path / "features/.mltool/features/base/train.parquet", index=False)
    assert finalize_project(path) == 2
    assert "stale" in capsys.readouterr().out

    # raw dataset changed after tuning
    path, _ = final_project(tmp_path / "data")
    (tmp_path / "data/data/dataset.csv").write_text("x,z,label\n1,1,0\n2,2,1\n")
    assert finalize_project(path) == 2
    assert "stale" in capsys.readouterr().out


def test_selected_json_tampering_is_detected(tmp_path: Path) -> None:
    path, _ = final_project(tmp_path)
    selected_path = tmp_path / ".mltool/tuning/selected.json"
    selected = json.loads(selected_path.read_text())
    selected["feature_artifacts"]["train_sha256"] = "0" * 64
    selected_path.write_text(json.dumps(selected))
    with pytest.raises(FinalizeError, match="re-materialized after tuning"):
        run_finalize(path)


# --- final-result ---------------------------------------------------------------------


def test_final_result_is_read_only_and_warns_on_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path, _ = final_project(tmp_path)
    assert final_result_project(path) == 2
    assert 'run "mltool finalize" first' in capsys.readouterr().out
    run_finalize(path)
    before = (tmp_path / ".mltool/final/result.json").read_bytes()
    monkeypatch.setattr(
        "mltool.cli.finalize_experiment", lambda *a, **k: pytest.fail("must not refit")
    )
    assert final_result_project(path) == 0
    out = capsys.readouterr().out
    assert "MLTool final result" in out and "roc_auc=" in out and "Warnings" not in out
    assert (tmp_path / ".mltool/final/result.json").read_bytes() == before
    assert load_persisted_final(load_config(path)).warning is None

    raw = yaml.safe_load(path.read_text())
    raw["models"] = [{"name": "only", "family": "GBM", "params": {}}]
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert load_persisted_final(load_config(path)).warning is not None


def test_final_result_warns_when_tuning_selection_changed(tmp_path: Path) -> None:
    path, _ = final_project(tmp_path)
    run_finalize(path)
    selected_path = tmp_path / ".mltool/tuning/selected.json"
    selected = json.loads(selected_path.read_text())
    selected["best_hyperparameters"] = {"learning_rate": 0.9}
    selected_path.write_text(json.dumps(selected))
    assert "tuning selection differs" in load_persisted_final(load_config(path)).warning


def test_finalize_cli_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, _ = final_project(tmp_path)
    monkeypatch.setattr(
        "mltool.cli.finalize_experiment",
        lambda plan, finalize_input: SimpleNamespace(render=lambda: "ok"),
    )
    assert finalize_project(path) == 0


# --- persisted refit transform state -------------------------------------------------


def test_refit_preprocessor_and_plugin_state_are_persisted_and_reproduce_the_fit_transform(
    tmp_path: Path,
) -> None:
    import cloudpickle

    from mltool.data import load_dataset
    from mltool.splitting import split_dataset

    path, audits = final_project(tmp_path, plugins=True)
    FinalAdapter.calls = []
    run_finalize(path)
    final = tmp_path / ".mltool/final"
    selected = json.loads((tmp_path / ".mltool/tuning/selected.json").read_text())
    chosen = "a" if selected["feature_set"] == "with_a" else "b"
    column = f"x{chosen}"

    assert (final / "preprocessor.pkl").is_file()
    assert (final / "feature_plugins" / f"{chosen}.pkl").is_file()
    assert not (final / "feature_plugins" / f"{'b' if chosen == 'a' else 'a'}.pkl").exists()
    result = json.loads((final / "result.json").read_text())
    manifest = json.loads((final / "manifest.json").read_text())
    # relative to .mltool/final, so they resolve in every registry copy too
    assert result["refit"]["preprocessor"]["artifact"] == "preprocessor.pkl"
    assert result["refit"]["feature_plugins"][0]["artifact"] == f"feature_plugins/{chosen}.pkl"
    assert manifest["artifacts"] == {
        "predictor": "predictor",
        "preprocessor": "preprocessor.pkl",
        "feature_plugins": {chosen: f"feature_plugins/{chosen}.pkl"},
    }

    with (final / "preprocessor.pkl").open("rb") as stream:
        preprocessor = cloudpickle.load(stream)
    with (final / "feature_plugins" / f"{chosen}.pkl").open("rb") as stream:
        plugin = cloudpickle.load(stream)

    config = load_config(path)
    splits = split_dataset(
        load_dataset(config.data).frame, target="label", task_type="binary", config=config.split
    )
    combined = pd.concat([splits.train, splits.validation])
    # both were refit on train+validation, not train alone
    assert preprocessor.mean == pytest.approx(combined["x"].mean())
    assert plugin.mean == pytest.approx(combined["x"].mean())
    assert plugin.mean != pytest.approx(splits.train["x"].mean())

    # Reproduce the fit-time transform on held-out (non-test) rows.
    sample = splits.validation.sample(6, random_state=0)
    features = preprocessor.transform(sample.drop(columns=["label"]))
    rebuilt = pd.concat([features, plugin.transform(features)], axis=1)
    fitted = FinalAdapter.calls[0]["train_data"].loc[sample.index]
    assert rebuilt.columns.tolist() == [c for c in fitted.columns if c != "label"]
    pd.testing.assert_frame_equal(rebuilt, fitted.drop(columns=["label"]))
    assert column in rebuilt
    # and never touched the test rows
    assert set(sample.index).isdisjoint(splits.test.index)


def test_no_preprocessor_or_plugins_means_no_state_files(tmp_path: Path) -> None:
    path, _ = final_project(tmp_path)
    run_finalize(path)
    final = tmp_path / ".mltool/final"
    assert not (final / "preprocessor.pkl").exists()
    assert not (final / "feature_plugins").exists()
    manifest = json.loads((final / "manifest.json").read_text())
    assert manifest["artifacts"]["preprocessor"] is None
    assert manifest["artifacts"]["feature_plugins"] == {}


def test_unserializable_fitted_state_aborts_before_the_test_is_evaluated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = final_project(tmp_path, plugins=True)
    import mltool.finalize as finalize_module

    def boom(obj: Any, stream: Any) -> None:
        raise TypeError("cannot pickle")

    monkeypatch.setattr(finalize_module.cloudpickle, "dump", boom)
    FinalAdapter.calls = []
    with pytest.raises(FinalizeError, match="could not serialize refit preprocessor"):
        run_finalize(path)
    assert FinalAdapter.calls == []  # no fit, hence no test evaluation
    assert not (tmp_path / ".mltool/final").exists()
    assert not list((tmp_path / ".mltool").glob(".final-staging-*"))
