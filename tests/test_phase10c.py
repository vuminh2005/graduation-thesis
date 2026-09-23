"""Deterministic tie-break among tuned trials; training.seed makes train stale."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from mltool.autogluon_adapter import AutoGluonAdapter, AutoGluonOutput, break_ties_by_trial_number
from mltool.cli import leaderboard_project, tune_project
from mltool.config import HpoConfig, ModelConfig, TaskConfig, TrainingConfig, load_config
from mltool.experiment import build_experiment_plan
from mltool.reporting import render_status
from mltool.training import load_persisted_leaderboard
from mltool.tuning import TuningError, build_tuning_selection, tune_experiment
from test_audit_fixes import build, edit, fake_adapters  # noqa: F401  (autouse fixture)
from test_phase5 import TuningAdapter


class TrialPredictor:
    """AutoGluon-shaped HPO result: trials "LightGBM/T<n>" with val_scores.

    ``predict`` answers with the current best model's id, so a test can see which
    trial MLTool's validation scores came from.
    """

    scores: dict[str, float] = {}
    autogluon_pick = ""
    calls: list[tuple[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self._best = type(self).autogluon_pick

    def fit(self, **kwargs: Any) -> "TrialPredictor":
        type(self).calls.append(("fit", None))
        return self

    @property
    def model_best(self) -> str:
        return self._best

    def set_model_best(self, model: str, save_trainer: bool = False) -> None:
        type(self).calls.append(("set_model_best", (model, save_trainer)))
        self._best = model

    def model_names(self) -> list[str]:
        return list(type(self).scores)

    def info(self) -> dict[str, Any]:
        return {"model_info": {
            name: {"val_score": score, "hyperparameters": {"trial": name, "seed": 7}}
            for name, score in type(self).scores.items()
        }}

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        type(self).calls.append(("predict", self._best))
        return pd.Series(["yes"] * len(frame), index=frame.index)

    def predict_proba(self, frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({"no": [0.2] * len(frame), "yes": [0.8] * len(frame)}, index=frame.index)


def tune_fit(tmp_path: Path, scores: dict[str, float], pick: str, hpo: bool = True) -> AutoGluonOutput:
    TrialPredictor.scores, TrialPredictor.autogluon_pick, TrialPredictor.calls = scores, pick, []
    adapter = AutoGluonAdapter(predictor_factory=TrialPredictor, version_resolver=lambda: "t")
    train = pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]})
    return adapter.fit_predict(
        train_data=train, validation_features=pd.DataFrame({"x": [4, 5]}),
        task=TaskConfig("binary", "label", "yes"), model=ModelConfig("gbm", "GBM"),
        primary_metric="roc_auc", predictor_path=tmp_path / "p",
        training=TrainingConfig(seed=7), hpo=HpoConfig(2, 4, 30) if hpo else None,
    )


# =========================== tie-break ===========================================


def test_a_tie_goes_to_the_lowest_trial_number(tmp_path: Path) -> None:
    scores = {"LightGBM/T1": 0.80, "LightGBM/T2": 0.90, "LightGBM/T3": 0.90, "LightGBM/T4": 0.85}
    output = tune_fit(tmp_path, scores, pick="LightGBM/T3")  # AutoGluon's timing-based pick
    assert output.best_model == "LightGBM/T2"
    assert output.best_hyperparameters == {"trial": "LightGBM/T2", "seed": 7}
    assert output.tied_trials == ["LightGBM/T2", "LightGBM/T3"]
    assert output.autogluon_best_trial == "LightGBM/T3"
    # persisted, and applied before MLTool predicts the validation split
    assert TrialPredictor.calls[1] == ("set_model_best", ("LightGBM/T2", True))
    assert ("predict", "LightGBM/T2") in TrialPredictor.calls
    assert ("predict", "LightGBM/T3") not in TrialPredictor.calls


def test_trial_numbers_compare_as_numbers_not_text(tmp_path: Path) -> None:
    scores = {"LightGBM/T10": 0.9, "LightGBM/T2": 0.9, "LightGBM/T9": 0.5}
    assert tune_fit(tmp_path, scores, pick="LightGBM/T10").best_model == "LightGBM/T2"


def test_a_tie_already_on_the_lowest_trial_is_still_recorded(tmp_path: Path) -> None:
    output = tune_fit(tmp_path, {"LightGBM/T1": 0.9, "LightGBM/T2": 0.9}, pick="LightGBM/T1")
    assert output.best_model == "LightGBM/T1" and output.tied_trials == ["LightGBM/T1", "LightGBM/T2"]


def test_no_tie_leaves_autogluons_choice_untouched(tmp_path: Path) -> None:
    scores = {"LightGBM/T1": 0.80, "LightGBM/T2": 0.90, "LightGBM/T3": 0.89}
    output = tune_fit(tmp_path, scores, pick="LightGBM/T2")
    assert output.best_model == "LightGBM/T2" and output.tied_trials == []
    assert not any(call[0] == "set_model_best" for call in TrialPredictor.calls)


def test_a_tie_among_worse_trials_does_not_count(tmp_path: Path) -> None:
    scores = {"LightGBM/T1": 0.7, "LightGBM/T2": 0.7, "LightGBM/T3": 0.9}
    output = tune_fit(tmp_path, scores, pick="LightGBM/T3")
    assert output.best_model == "LightGBM/T3" and output.tied_trials == []


def test_models_without_trial_names_are_ignored() -> None:
    class Single(TrialPredictor):
        pass

    Single.scores, Single.autogluon_pick, Single.calls = {"LightGBM": 0.9, "Other": 0.9}, "LightGBM", []
    assert break_ties_by_trial_number(Single()) == ([], "LightGBM")
    assert Single.calls == []


def test_training_without_hpo_never_breaks_ties(tmp_path: Path) -> None:
    class Plain(TrialPredictor):
        def info(self) -> dict[str, Any]:  # train never reads it
            raise AssertionError("no tie-break without HPO")

    Plain.scores, Plain.autogluon_pick, Plain.calls = {"LightGBM/T1": 0.9, "LightGBM/T2": 0.9}, "LightGBM/T2", []
    adapter = AutoGluonAdapter(predictor_factory=Plain, version_resolver=lambda: "t")
    output = adapter.fit_predict(
        train_data=pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]}),
        validation_features=pd.DataFrame({"x": [4, 5]}), task=TaskConfig("binary", "label", "yes"),
        model=ModelConfig("gbm", "GBM"), primary_metric="roc_auc", predictor_path=tmp_path / "p",
        training=TrainingConfig(seed=7), hpo=None,
    )
    assert output.tied_trials == [] and not any(c[0] == "set_model_best" for c in Plain.calls)


class TiedTuningAdapter(TuningAdapter):
    def fit_predict(self, **kwargs: Any) -> AutoGluonOutput:
        output = super().fit_predict(**kwargs)
        if kwargs.get("hpo") is not None:
            output.tied_trials = ["LightGBM/T2", "LightGBM/T5"]
            output.autogluon_best_trial = "LightGBM/T5"
        return output


def test_the_tuning_result_records_the_tie_break(tmp_path: Path) -> None:
    path = build(tmp_path, through="train")  # GBM + RF; RF is carried over
    plan = build_experiment_plan(load_config(path))
    tune_experiment(plan, build_tuning_selection(plan), adapter_factory=TiedTuningAdapter)
    root = tmp_path / ".mltool/tuning/candidates"
    gbm = json.loads((root / "base__lightgbm/result.json").read_text())
    assert gbm["tie_break_applied"] is True
    assert gbm["tied_trials"] == ["LightGBM/T2", "LightGBM/T5"]
    assert gbm["autogluon_best_trial"] == "LightGBM/T5"
    rf = json.loads((root / "base__forest/result.json").read_text())
    assert rf["tie_break_applied"] is False and rf["tied_trials"] == []


def test_no_tie_is_recorded_as_false(tmp_path: Path) -> None:
    path = build(tmp_path, through="tune")
    gbm = json.loads((tmp_path / ".mltool/tuning/candidates/base__lightgbm/result.json").read_text())
    assert gbm["tie_break_applied"] is False and gbm["tied_trials"] == []


# =========================== training.seed staleness =============================


def train_state(path: Path) -> str:
    return next(line.split()[1] for line in render_status(load_config(path)).splitlines()
                if line.startswith("train"))


def test_an_unchanged_seed_keeps_train_fresh(tmp_path: Path) -> None:
    path = build(tmp_path, through="train")
    config = load_config(path)
    assert load_persisted_leaderboard(config).warning is None
    build_tuning_selection(build_experiment_plan(config))
    assert train_state(path) == "fresh"


def test_a_seed_change_makes_train_stale_everywhere(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = build(tmp_path, through="train")
    edit(path, lambda raw: raw["training"].__setitem__("seed", 8))
    assert train_state(path) == "stale"
    capsys.readouterr()
    assert tune_project(path) == 2
    assert '"training.seed" differs from the current config' in capsys.readouterr().out
    assert leaderboard_project(path) == 0
    assert "training.seed" in capsys.readouterr().out


@pytest.mark.parametrize(("configured", "stale"), [(None, False), (7, True)])
def test_a_manifest_from_before_the_seed_was_recorded(
    tmp_path: Path, configured: int | None, stale: bool
) -> None:
    path = build(tmp_path, through="train")
    manifest_path = tmp_path / ".mltool/training/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["seed"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    edit(path, lambda raw: raw["training"].__setitem__("seed", configured))
    config = load_config(path)
    if stale:
        with pytest.raises(TuningError, match='"training.seed" differs'):
            build_tuning_selection(build_experiment_plan(config))
    else:
        # effective seeds are recorded per candidate too; with no seed they are None
        assert load_persisted_leaderboard(config).warning is None
