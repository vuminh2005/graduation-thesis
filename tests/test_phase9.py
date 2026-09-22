"""Phase 9: cross-validated candidate evaluation and FeatureSet plugin_inputs.

Fake-predictor style, matching test_phase4..8. The leakage test is the important
one: a recording preprocessor and a recording plugin write down the row labels
handed to their fit(), and every fold must have seen only its own training rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from mltool.autogluon_adapter import AutoGluonOutput
from mltool.config import ConfigError, CvConfig, FeatureSetConfig, ModelConfig, load_config
from mltool.cross_validation import (
    CrossValidationError,
    build_cv_plan,
    cost_warning,
    repeat_seed,
)
from mltool.experiment import ExperimentError, build_experiment_plan, load_feature_artifacts
from mltool.feature_materialization import (
    FeatureMaterializationError,
    build_feature_set_from_frames,
    materialize_feature_sets,
)
from mltool.finalize import finalize_experiment, load_finalize_input
from mltool.training import build_leaderboard, cv_signature, train_experiment
from mltool.tuning import (
    TuningError,
    build_tuning_selection,
    tuned_model_config,
    tune_experiment,
)
from test_phase4 import materialize, project_config, write_project
from test_phase5 import hpo_raw
from test_phase6 import FinalAdapter

MODELS = [
    {"name": "gbm", "family": "GBM", "params": {}},
    {"name": "rf", "family": "RF", "params": {}},
]

# A recording preprocessor and plugin: each fit() appends the row labels it saw.
RECORDER_SOURCE = '''
import json, pathlib
import pandas as pd

LOG = pathlib.Path(__file__).with_name("fit_log.jsonl")


def _record(who, X):
    with LOG.open("a") as stream:
        stream.write(json.dumps(
            {"who": who, "rows": sorted(int(i) for i in X.index),
             "columns": list(X.columns)}) + "\\n")


class RecordingPre:
    def fit(self, X):
        _record("preprocessor", X)
        self.mean = float(X["x"].mean())
        return self

    def transform(self, X):
        out = X.copy()
        out["x"] = out["x"] - self.mean
        return out


class RecordingPlugin:
    def fit(self, X):
        _record("plugin", X)
        self.mean = float(X["z"].mean())
        return self

    def transform(self, X):
        return pd.DataFrame({"z_centered": X["z"] - self.mean}, index=X.index)
'''


class CvFakeAdapter:
    """Records every fit_predict call; scores from the first numeric column."""

    calls: list[dict[str, Any]] = []

    def fit_predict(self, **kwargs: Any) -> AutoGluonOutput:
        type(self).calls.append(kwargs)
        features = kwargs["validation_features"]
        predictor_path = kwargs["predictor_path"]
        predictor_path.mkdir(parents=True)
        (predictor_path / "fake.txt").write_text("fake", encoding="utf-8")

        numeric = features.select_dtypes("number")
        raw = numeric.iloc[:, 0] if numeric.shape[1] else pd.Series(0.5, index=features.index)
        spread = float(raw.max() - raw.min())
        scores = (raw - raw.min()) / spread if spread else pd.Series(0.5, index=features.index)
        if kwargs["task"].type == "regression":
            predictions, probabilities, positive = raw.astype(float), None, None
        else:
            predictions = (scores > 0.5).astype(int)
            probabilities = pd.DataFrame({0: 1 - scores, 1: scores}, index=features.index)
            positive = 1
        best_model = best_hyperparameters = None
        if kwargs.get("hpo") is not None:
            best_model = "LightGBM/T1"
            best_hyperparameters = {"learning_rate": 0.05, "seed": 7}
        elif kwargs["model"].family == "ENSEMBLE":
            best_model = "WeightedEnsemble_L2"
            best_hyperparameters = dict(kwargs["model"].params)
        return AutoGluonOutput(
            predictions=predictions,
            probabilities=probabilities,
            positive_class=positive,
            trained_models=["LightGBM"],
            autogluon_version="1.6.fake",
            best_model=best_model,
            best_hyperparameters=best_hyperparameters,
        )


def cv_project(
    root: Path,
    *,
    cv: dict[str, Any] | None = None,
    features: dict[str, Any] | None = None,
    models: list[dict[str, Any]] | None = None,
    recorder: bool = False,
    frame: pd.DataFrame | None = None,
) -> Path:
    evaluation: dict[str, Any] = {"primary_metric": "roc_auc", "secondary_metrics": ["accuracy"]}
    if cv is not None:
        evaluation["cv"] = cv
    raw = project_config(models=models or MODELS, evaluation=evaluation, features=features)
    if recorder:
        raw["preprocessing"]["external"] = {
            "enabled": True, "entrypoint": "./rec.py:RecordingPre", "params": {},
        }
    root.mkdir(parents=True, exist_ok=True)
    (root / "rec.py").write_text(RECORDER_SOURCE, encoding="utf-8")
    return materialize(root, raw=raw, frame=frame)


def run_train(path: Path) -> Any:
    CvFakeAdapter.calls = []
    return train_experiment(
        build_experiment_plan(load_config(path)), adapter_factory=CvFakeAdapter
    )


# =========================== config validation ================================


def test_cv_config_defaults_and_parsing(tmp_path: Path) -> None:
    path = write_project(tmp_path, raw=project_config(models=MODELS))
    assert load_config(path).evaluation.cv is None  # absent stays holdout

    def with_cv(cv: Any) -> Any:
        raw = yaml.safe_load(path.read_text())
        raw["evaluation"] = {"primary_metric": "roc_auc", "secondary_metrics": ["f1"], "cv": cv}
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        return load_config(path)

    assert with_cv({}).evaluation.cv == CvConfig(folds=5, repeats=1)
    assert with_cv({"folds": 4, "repeats": 3}).evaluation.cv == CvConfig(folds=4, repeats=3)
    assert with_cv(None).evaluation.cv is None
    assert CvConfig(4, 3).total_fits == 12


@pytest.mark.parametrize(
    ("cv", "message"),
    [
        ({"folds": 1}, "at least 2"),
        ({"folds": 0}, "at least 2"),
        ({"folds": True}, "at least 2"),
        ({"folds": 2.5}, "at least 2"),
        ({"repeats": 0}, "at least 1"),
        ({"repeats": True}, "at least 1"),
        ({"fold": 5}, 'unsupported "evaluation.cv" setting'),
        ({"folds": 5, "seed": 1}, 'unsupported "evaluation.cv" setting'),
        ([], '"evaluation.cv" must be a mapping'),
    ],
)
def test_invalid_cv_config_is_rejected(tmp_path: Path, cv: Any, message: str) -> None:
    raw = project_config(
        models=MODELS,
        evaluation={"primary_metric": "roc_auc", "secondary_metrics": ["f1"], "cv": cv},
    )
    with pytest.raises(ConfigError, match=message):
        load_config(write_project(tmp_path, raw=raw))


def test_unknown_evaluation_key_still_rejected(tmp_path: Path) -> None:
    raw = project_config(
        models=MODELS, evaluation={"primary_metric": "roc_auc", "cvv": {"folds": 3}}
    )
    with pytest.raises(ConfigError, match='unsupported "evaluation" setting'):
        load_config(write_project(tmp_path, raw=raw))


def plugin_set(**extra: Any) -> dict[str, Any]:
    base = {"name": "s", "source_columns": ["x"], "plugins": ["p"]}
    base.update(extra)
    return {
        "plugins": [{"name": "p", "entrypoint": "./rec.py:RecordingPlugin", "params": {}}],
        "sets": [base],
    }


def test_plugin_inputs_parsing(tmp_path: Path) -> None:
    raw = project_config(models=MODELS, features=plugin_set(plugin_inputs=["z"]))
    config = load_config(write_project(tmp_path, raw=raw))
    assert config.features.sets[0].plugin_inputs == ["z"]
    # absent is exactly today's behavior
    plain = load_config(
        write_project(tmp_path / "plain", raw=project_config(models=MODELS, features=plugin_set()))
    )
    assert plain.features.sets[0].plugin_inputs == []
    assert FeatureSetConfig("s", ["*"], []).plugin_inputs == []


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (plugin_set(plugin_inputs="z"), "list of non-empty strings"),
        (plugin_set(plugin_inputs=[""]), "list of non-empty strings"),
        (plugin_set(plugin_inputs=["z", "z"]), "unique"),
        (plugin_set(plugin_inputs=["*"]), 'may not contain "\\*"'),
        (
            {"plugins": [], "sets": [{"name": "s", "source_columns": ["x"], "plugin_inputs": ["z"]}]},
            "requires at least one plugin",
        ),
        (plugin_set(plugin_input=["z"]), 'unsupported "features.sets\\[0\\]" setting'),
    ],
)
def test_invalid_plugin_inputs_are_rejected(
    tmp_path: Path, features: dict[str, Any], message: str
) -> None:
    raw = project_config(models=MODELS, features=features)
    with pytest.raises(ConfigError, match=message):
        load_config(write_project(tmp_path, raw=raw))


# =========================== no cv => Phase 8 behavior ========================


def test_without_cv_everything_matches_phase8(tmp_path: Path) -> None:
    path = cv_project(tmp_path)
    result = run_train(path)
    assert result.cv is None
    assert len(CvFakeAdapter.calls) == len(MODELS)  # one fit per candidate, not per fold
    # Phase 8 passed no hpo kwarg at all on this path; that must stay true.
    assert all("hpo" not in call for call in CvFakeAdapter.calls)

    stored = json.loads((tmp_path / ".mltool/training/candidates/base__gbm/result.json").read_text())
    assert "cv" not in stored and "predictor_persisted" not in stored
    assert stored["predictor_path"].endswith("candidates/base__gbm/predictor")
    assert (tmp_path / ".mltool/training/candidates/base__gbm/predictor/fake.txt").is_file()
    manifest = json.loads((tmp_path / ".mltool/training/manifest.json").read_text())
    assert "cv" not in manifest
    leaderboard = json.loads((tmp_path / ".mltool/training/leaderboard.json").read_text())
    assert all("primary_score_std" not in row and "folds" not in row for row in leaderboard)
    assert "Cross-validation" not in result.render()
    assert cv_signature(load_config(path)) is None


# =========================== fold plan ========================================


def test_fold_plan_is_deterministic_and_covers_the_development_set(tmp_path: Path) -> None:
    path = cv_project(tmp_path, cv={"folds": 4, "repeats": 2})
    config = load_config(path)
    first, second = build_cv_plan(config), build_cv_plan(config)
    assert first.fingerprint == second.fingerprint
    assert [f.holdout_labels for f in first.folds] == [f.holdout_labels for f in second.folds]
    assert len(first.folds) == 8 and first.total_fits == 8

    prepared = json.loads((tmp_path / ".mltool/prepared/manifest.json").read_text())["split"]
    assert len(first.development) == prepared["train_rows"] + prepared["validation_rows"]
    assert len(first.test_labels) == prepared["test_rows"]
    assert not set(first.development.index) & set(first.test_labels)

    for repeat in range(2):
        folds = [f for f in first.folds if f.repeat == repeat]
        holdout = [label for fold in folds for label in fold.holdout_labels]
        assert sorted(holdout) == sorted(first.development.index)  # a partition
        assert len(holdout) == len(set(holdout))
        for fold in folds:
            assert not set(fold.train_labels) & set(fold.holdout_labels)
            assert set(fold.train_labels) | set(fold.holdout_labels) == set(first.development.index)


def test_repeats_use_different_but_derived_seeds(tmp_path: Path) -> None:
    path = cv_project(tmp_path, cv={"folds": 3, "repeats": 3})
    plan = build_cv_plan(load_config(path))
    seeds = sorted({fold.seed for fold in plan.folds})
    assert len(seeds) == 3
    assert seeds == sorted({repeat_seed(42, r) for r in range(3)})
    first = {f.fold: f.holdout_labels for f in plan.folds if f.repeat == 0}
    second = {f.fold: f.holdout_labels for f in plan.folds if f.repeat == 1}
    assert first != second  # a different partition, not a re-run of the same one


def test_a_different_split_seed_changes_the_fold_fingerprint(tmp_path: Path) -> None:
    path = cv_project(tmp_path, cv={"folds": 4, "repeats": 1})
    before = build_cv_plan(load_config(path)).fingerprint
    raw = yaml.safe_load(path.read_text())
    raw["split"]["random_seed"] = 7
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    # re-prepare so the development rows match the new seed
    from mltool.cli import prepare_project

    assert prepare_project(path) == 0
    assert build_cv_plan(load_config(path)).fingerprint != before


def test_classification_folds_are_stratified(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"x": range(200), "z": [v % 7 for v in range(200)],
         "label": [1 if v % 5 == 0 else 0 for v in range(200)]}  # 20% positives
    )
    path = cv_project(tmp_path, cv={"folds": 5, "repeats": 1}, frame=frame)
    plan = build_cv_plan(load_config(path))
    assert plan.stratified is True
    target = plan.development["label"]
    overall = target.mean()
    for fold in plan.folds:
        held = target.loc[list(fold.holdout_labels)].mean()
        assert abs(held - overall) < 0.05


def test_regression_folds_are_not_stratified(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": range(120), "z": [v % 7 for v in range(120)],
                          "label": [float(v) * 1.5 for v in range(120)]})
    raw = project_config(
        task="regression", positive_class=None, models=MODELS,
        evaluation={"primary_metric": "rmse", "secondary_metrics": ["mae"],
                    "cv": {"folds": 4, "repeats": 1}},
    )
    path = materialize(tmp_path, raw=raw, frame=frame)
    plan = build_cv_plan(load_config(path))
    assert plan.stratified is False and len(plan.folds) == 4


def test_too_many_folds_is_a_clear_error(tmp_path: Path) -> None:
    path = cv_project(tmp_path, cv={"folds": 200, "repeats": 1})
    with pytest.raises(CrossValidationError, match="cannot build 200 stratified folds"):
        build_cv_plan(load_config(path))


def test_cost_warning_states_the_number_of_fits(tmp_path: Path) -> None:
    path = cv_project(tmp_path, cv={"folds": 4, "repeats": 2})
    plan = build_cv_plan(load_config(path))
    assert cost_warning(3, plan, "train") == (
        "MLTool train: cross-validation will fit 3 candidate(s) x 4 folds x "
        "2 repeat(s) = 24 models, sequentially."
    )


# =========================== the leakage guarantee ============================


def read_fit_log(root: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (root / "fit_log.jsonl").read_text().splitlines()]


def test_every_fold_refits_on_its_own_training_rows_only(tmp_path: Path) -> None:
    """No fold may fit on its held-out rows, and nothing may touch the test rows."""
    features = {
        "plugins": [{"name": "p", "entrypoint": "./rec.py:RecordingPlugin", "params": {}}],
        "sets": [{"name": "base", "source_columns": ["x", "z"], "plugins": ["p"]}],
    }
    path = cv_project(
        tmp_path, cv={"folds": 4, "repeats": 2}, features=features, recorder=True,
        models=[{"name": "gbm", "family": "GBM", "params": {}}],
    )
    (tmp_path / "fit_log.jsonl").unlink(missing_ok=True)  # drop the materialization fits
    plan = build_cv_plan(load_config(path))
    run_train(path)

    records = read_fit_log(tmp_path)
    assert len(records) == 2 * len(plan.folds)  # one preprocessor + one plugin per fold
    test_rows = {int(label) for label in plan.test_labels}
    assert test_rows

    for index, fold in enumerate(plan.folds):
        expected = {int(label) for label in fold.train_labels}
        holdout = {int(label) for label in fold.holdout_labels}
        pair = records[2 * index: 2 * index + 2]
        assert [entry["who"] for entry in pair] == ["preprocessor", "plugin"]
        for entry in pair:
            seen = set(entry["rows"])
            assert seen == expected, f"fold {index} {entry['who']} saw the wrong rows"
            assert not seen & holdout, f"fold {index} {entry['who']} saw held-out rows"
            assert not seen & test_rows, f"fold {index} {entry['who']} saw test rows"

    every_row_seen = set().union(*(set(entry["rows"]) for entry in records))
    assert not every_row_seen & test_rows
    # and the candidate itself was only ever fitted on fold-training rows
    for call, fold in zip(CvFakeAdapter.calls, plan.folds, strict=True):
        assert set(call["train_data"].index) == set(fold.train_labels)
        assert set(call["validation_features"].index) == set(fold.holdout_labels)
        assert not set(call["validation_features"].index) & test_rows


def test_cv_does_not_read_the_materialized_feature_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Their plugins were fitted on the whole train split, which overlaps the folds."""
    path = cv_project(tmp_path, cv={"folds": 3, "repeats": 1})
    plan = build_experiment_plan(load_config(path))  # plan validation reads them; scoring must not
    opened: list[str] = []
    real_read = pd.read_parquet

    def guarded(path_arg: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        opened.append(str(path_arg))
        return real_read(path_arg, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded)
    CvFakeAdapter.calls = []
    train_experiment(plan, adapter_factory=CvFakeAdapter)
    assert not [name for name in opened if ".mltool/features/" in name]
    assert not [name for name in opened if "test.parquet" in name]


def test_training_manifest_still_says_test_data_unused(tmp_path: Path) -> None:
    path = cv_project(tmp_path, cv={"folds": 3, "repeats": 1})
    run_train(path)
    manifest = json.loads((tmp_path / ".mltool/training/manifest.json").read_text())
    assert manifest["test_data_used"] is False
    assert manifest["cv"]["folds"] == 3 and manifest["cv"]["repeats"] == 1
    assert manifest["cv"]["fold_fingerprint"]


# =========================== train in cv mode =================================


def test_train_scores_every_fold_and_persists_them(tmp_path: Path) -> None:
    path = cv_project(tmp_path, cv={"folds": 4, "repeats": 2})
    result = run_train(path)
    assert result.is_successful
    assert len(CvFakeAdapter.calls) == 8 * len(MODELS)
    assert all(call["hpo"] is None for call in CvFakeAdapter.calls)

    stored = json.loads((tmp_path / ".mltool/training/candidates/base__gbm/result.json").read_text())
    folds = stored["cv"]["fold_metrics"]
    assert len(folds) == 8
    assert {entry["repeat"] for entry in folds} == {0, 1}
    assert {entry["fold"] for entry in folds} == {0, 1, 2, 3}
    assert all(entry["rows"]["train"] + entry["rows"]["holdout"] == 85 for entry in folds)
    mean = sum(entry["metrics"]["roc_auc"] for entry in folds) / len(folds)
    assert stored["metrics"]["roc_auc"] == pytest.approx(mean)
    assert stored["cv"]["metric_std"]["roc_auc"] >= 0
    # fold predictors are scratch; nothing downstream loads a training predictor
    assert stored["predictor_persisted"] is False and stored["predictor_path"] is None
    assert not (tmp_path / ".mltool/training/candidates/base__gbm/predictor").exists()

    out = result.render()
    assert "Cross-validation" in out and "4 folds x 2 repeat(s) = 8 fits per candidate" in out
    assert "+/-" in out and "(8 folds)" in out


def test_leaderboard_ranks_by_cv_mean_in_both_directions() -> None:
    def candidate(name: str, values: list[float], metric: str) -> dict[str, Any]:
        mean = sum(values) / len(values)
        return {
            "candidate_id": name, "feature_set": "base", "status": "SUCCEEDED",
            "model": {"name": name, "family": "GBM", "params": {}},
            "metrics": {metric: mean}, "training_seconds": 1.0,
            "cv": {"metric_std": {metric: 0.25}, "total_fits_per_candidate": len(values)},
        }

    results = [candidate("a", [0.5, 0.9], "roc_auc"), candidate("b", [0.8, 1.0], "roc_auc"),
               candidate("c", [0.6, 0.6], "roc_auc")]
    board = build_leaderboard(results, primary_metric="roc_auc", direction="maximize")
    assert [row["candidate_id"] for row in board] == ["b", "a", "c"]
    assert board[0]["primary_score"] == pytest.approx(0.9)
    assert board[0]["primary_score_std"] == 0.25 and board[0]["folds"] == 2

    losses = [candidate("a", [0.5, 0.9], "rmse"), candidate("b", [0.8, 1.0], "rmse"),
              candidate("c", [0.6, 0.6], "rmse")]
    board = build_leaderboard(losses, primary_metric="rmse", direction="minimize")
    assert [row["candidate_id"] for row in board] == ["c", "a", "b"]


def test_cv_leaderboard_primary_score_is_the_fold_mean(tmp_path: Path) -> None:
    path = cv_project(tmp_path, cv={"folds": 3, "repeats": 1})
    result = run_train(path)
    for row in result.leaderboard:
        stored = json.loads(
            (tmp_path / f".mltool/training/candidates/{row['candidate_id']}/result.json").read_text()
        )
        folds = stored["cv"]["fold_metrics"]
        assert row["primary_score"] == pytest.approx(
            sum(f["metrics"]["roc_auc"] for f in folds) / len(folds)
        )
        assert row["folds"] == 3


# =========================== tune in cv mode ==================================


def test_tune_cv_scores_the_tuned_configuration_without_hpo(tmp_path: Path) -> None:
    path = cv_project(tmp_path, cv={"folds": 3, "repeats": 1})
    hpo_raw(path, top_n=1)
    run_train(path)
    plan = build_experiment_plan(load_config(path))
    selection = build_tuning_selection(plan)
    CvFakeAdapter.calls = []
    result = tune_experiment(plan, selection, adapter_factory=CvFakeAdapter)

    searched = [call for call in CvFakeAdapter.calls if call["hpo"] is not None]
    cv_calls = [call for call in CvFakeAdapter.calls if call["hpo"] is None]
    assert len(searched) == 1  # the HPO search, on the train split, as before
    assert len(cv_calls) == 3  # one per fold, never with HPO
    # the CV fits use the searched hyperparameters, fixed
    assert all(call["model"].params == {"learning_rate": 0.05, "seed": 7} for call in cv_calls)

    stored = json.loads((tmp_path / ".mltool/tuning/candidates/base__gbm/result.json").read_text())
    assert len(stored["cv"]["fold_metrics"]) == 3
    assert stored["metrics"]["roc_auc"] == pytest.approx(
        sum(f["metrics"]["roc_auc"] for f in stored["cv"]["fold_metrics"]) / 3
    )
    assert stored["holdout_metrics"]["roc_auc"] >= 0  # kept for transparency
    selected = json.loads((tmp_path / ".mltool/tuning/selected.json").read_text())
    assert selected["tuned_validation_score"] == pytest.approx(stored["metrics"]["roc_auc"])
    manifest = json.loads((tmp_path / ".mltool/tuning/manifest.json").read_text())
    assert manifest["cv"]["folds"] == 3 and manifest["test_data_used"] is False
    assert "Cross-validation" in result.render()


def test_untunable_families_are_cv_scored_as_configured() -> None:
    from mltool.experiment import CandidateSpec, FeatureSetArtifact

    artifact = FeatureSetArtifact(
        name="base", path=Path("."), train_path=Path("."), validation_path=Path("."),
        manifest_path=Path("."), manifest={}, train_sha256="a", validation_sha256="b",
        feature_columns=["x"], train_rows=1, validation_rows=1,
    )
    rf = CandidateSpec("base__rf", artifact, ModelConfig("rf", "RF", {"n_estimators": 7}))
    # HPO did not apply: keep the candidate's own configuration
    assert tuned_model_config(rf, {"searched": 1}, hpo_effective=False) is rf.model
    # HPO applied: use the fixed winning hyperparameters
    tuned = tuned_model_config(rf, {"n_estimators": 42}, hpo_effective=True)
    assert tuned.params == {"n_estimators": 42} and tuned.family == "RF"
    assert tuned_model_config(rf, None, hpo_effective=True) is rf.model


def test_finalize_is_unchanged_by_cv(tmp_path: Path) -> None:
    path = cv_project(tmp_path, cv={"folds": 3, "repeats": 1})
    hpo_raw(path, top_n=1)
    run_train(path)
    plan = build_experiment_plan(load_config(path))
    tune_experiment(plan, build_tuning_selection(plan), adapter_factory=CvFakeAdapter)
    FinalAdapter.calls = []
    plan = build_experiment_plan(load_config(path))
    finalize_experiment(plan, load_finalize_input(plan), adapter_factory=FinalAdapter)
    assert len(FinalAdapter.calls) == 1  # still one refit and one test evaluation
    manifest = json.loads((tmp_path / ".mltool/final/manifest.json").read_text())
    assert manifest["test_data_used"] is True and manifest["test_evaluations"] == 1
    assert manifest["cv"] == {"folds": 3, "repeats": 1}


# =========================== staleness ========================================


def test_changing_cv_config_makes_training_stale(tmp_path: Path) -> None:
    path = cv_project(tmp_path, cv={"folds": 3, "repeats": 1})
    hpo_raw(path, top_n=1)
    run_train(path)
    raw = yaml.safe_load(path.read_text())
    raw["evaluation"]["cv"]["folds"] = 4
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    plan = build_experiment_plan(load_config(path))
    with pytest.raises(TuningError, match='"cv" differs from the current config'):
        build_tuning_selection(plan)


def test_adding_or_removing_cv_makes_training_stale(tmp_path: Path) -> None:
    path = cv_project(tmp_path)  # trained without cv
    hpo_raw(path, top_n=1)
    run_train(path)
    raw = yaml.safe_load(path.read_text())
    raw["evaluation"]["cv"] = {"folds": 3, "repeats": 1}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(TuningError, match='"cv" differs'):
        build_tuning_selection(build_experiment_plan(load_config(path)))


def test_changing_plugin_inputs_makes_features_stale(tmp_path: Path) -> None:
    features = {
        "plugins": [{"name": "p", "entrypoint": "./rec.py:RecordingPlugin", "params": {}}],
        "sets": [{"name": "base", "source_columns": ["x"], "plugin_inputs": ["z"], "plugins": ["p"]}],
    }
    path = cv_project(tmp_path, features=features)
    assert load_feature_artifacts(load_config(path))
    raw = yaml.safe_load(path.read_text())
    raw["features"]["sets"][0]["plugin_inputs"] = []
    raw["features"]["sets"][0]["source_columns"] = ["x", "z"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ExperimentError, match="plugin inputs changed|source columns changed"):
        load_feature_artifacts(load_config(path))


def test_feature_manifest_without_plugin_inputs_stays_fresh(tmp_path: Path) -> None:
    """Artifacts from before Phase 9 are equivalent to a set with no plugin_inputs."""
    path = cv_project(tmp_path)
    manifest_path = tmp_path / ".mltool/features/base/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["plugin_inputs"] == []
    del manifest["plugin_inputs"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_feature_artifacts(load_config(path))  # still fresh


# =========================== plugin_inputs ====================================


def replaced_features() -> dict[str, Any]:
    return {
        "plugins": [{"name": "p", "entrypoint": "./rec.py:RecordingPlugin", "params": {}}],
        "sets": [
            {"name": "kept", "source_columns": ["x", "z"], "plugins": ["p"]},
            {"name": "replaced", "source_columns": ["x"], "plugin_inputs": ["z"], "plugins": ["p"]},
        ],
    }


def test_plugin_inputs_feed_the_plugin_without_becoming_features(tmp_path: Path) -> None:
    path = cv_project(tmp_path, features=replaced_features())
    kept = json.loads((tmp_path / ".mltool/features/kept/manifest.json").read_text())
    replaced = json.loads((tmp_path / ".mltool/features/replaced/manifest.json").read_text())

    assert kept["final_feature_columns"] == ["x", "z", "z_centered"]
    assert replaced["final_feature_columns"] == ["x", "z_centered"]  # z is gone
    assert replaced["source_columns"] == ["x"] and replaced["plugin_inputs"] == ["z"]
    assert replaced["lineage"]["z_centered"] == {"source": "plugin", "plugin": "p"}
    assert "z" not in replaced["lineage"]

    frame = pd.read_parquet(tmp_path / ".mltool/features/replaced/train.parquet")
    assert list(frame.columns) == ["x", "z_centered", "label"]
    # the plugin really did read z
    records = [entry for entry in read_fit_log(tmp_path) if entry["who"] == "plugin"]
    assert ["x", "z"] in [entry["columns"] for entry in records]


def test_plugin_input_validation_at_materialization(tmp_path: Path) -> None:
    def build(**set_extra: Any) -> Path:
        features = {
            "plugins": [{"name": "p", "entrypoint": "./rec.py:RecordingPlugin", "params": {}}],
            "sets": [{"name": "s", "source_columns": ["x"], "plugins": ["p"], **set_extra}],
        }
        root = tmp_path / str(len(list(tmp_path.iterdir())))
        root.mkdir(parents=True)
        (root / "rec.py").write_text(RECORDER_SOURCE, encoding="utf-8")
        raw = project_config(models=MODELS, features=features)
        path = write_project(root, raw=raw)
        from mltool.cli import prepare_project

        assert prepare_project(path) == 0
        return path

    missing = build(plugin_inputs=["nope"])
    with pytest.raises(FeatureMaterializationError, match='missing plugin input "nope"'):
        materialize_feature_sets(load_config(missing))

    target = build(plugin_inputs=["label"])
    with pytest.raises(FeatureMaterializationError, match="may not use configured target column"):
        materialize_feature_sets(load_config(target))


def test_a_plugin_may_not_shadow_a_plugin_input(tmp_path: Path) -> None:
    (tmp_path / "shadow.py").write_text(
        "import pandas as pd\n"
        "class Shadow:\n"
        "    def fit(self, X): return self\n"
        "    def transform(self, X):\n"
        "        return pd.DataFrame({'z': X['z'] * 2}, index=X.index)\n",
        encoding="utf-8",
    )
    features = {
        "plugins": [{"name": "p", "entrypoint": "./shadow.py:Shadow", "params": {}}],
        "sets": [{"name": "s", "source_columns": ["x"], "plugin_inputs": ["z"], "plugins": ["p"]}],
    }
    raw = project_config(models=MODELS, features=features)
    path = write_project(tmp_path, raw=raw)
    from mltool.cli import prepare_project

    assert prepare_project(path) == 0
    with pytest.raises(FeatureMaterializationError, match='generated colliding column "z"'):
        materialize_feature_sets(load_config(path))


def test_plugin_inputs_are_honored_identically_in_cv_and_finalize(tmp_path: Path) -> None:
    features = {
        "plugins": [{"name": "p", "entrypoint": "./rec.py:RecordingPlugin", "params": {}}],
        "sets": [
            {"name": "base", "source_columns": ["x"], "plugin_inputs": ["z"], "plugins": ["p"]}
        ],
    }
    path = cv_project(
        tmp_path, cv={"folds": 3, "repeats": 1}, features=features,
        models=[{"name": "gbm", "family": "GBM", "params": {}}],
    )
    hpo_raw(path, top_n=1)
    expected = ["x", "z_centered"]

    run_train(path)  # CV: every fold's frames use the same recipe
    for call in CvFakeAdapter.calls:
        assert list(call["validation_features"].columns) == expected
        assert "z" not in call["train_data"].columns
        assert list(call["train_data"].columns) == [*expected, "label"]

    plan = build_experiment_plan(load_config(path))
    tune_experiment(plan, build_tuning_selection(plan), adapter_factory=CvFakeAdapter)
    FinalAdapter.calls = []
    plan = build_experiment_plan(load_config(path))
    finalize_experiment(plan, load_finalize_input(plan), adapter_factory=FinalAdapter)
    call = FinalAdapter.calls[0]
    assert list(call["test_features"].columns) == expected
    result = json.loads((tmp_path / ".mltool/final/result.json").read_text())
    assert result["refit"]["final_feature_columns"] == expected


def test_plugin_inputs_default_keeps_the_old_frame(tmp_path: Path) -> None:
    """A set without plugin_inputs behaves exactly as before."""
    config = load_config(
        write_project(
            tmp_path,
            raw=project_config(models=MODELS, features=plugin_set(source_columns=["x", "z"])),
        )
    )
    (tmp_path / "rec.py").write_text(RECORDER_SOURCE, encoding="utf-8")
    frames = {
        "fit": pd.DataFrame({"x": [1.0, 2, 3, 4], "z": [1, 2, 3, 4], "label": [0, 1, 0, 1]}),
        "out": pd.DataFrame({"x": [5.0, 6], "z": [5, 6], "label": [1, 0]}),
    }
    spec = config.features.sets[0]
    built = build_feature_set_from_frames(
        config, frames, "fit", spec, {p.name: p for p in config.features.plugins}
    )
    assert built.manifest["final_feature_columns"] == ["x", "z", "z_centered"]
    assert built.manifest["plugin_inputs"] == []
