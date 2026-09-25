from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype
import pytest
import yaml

from mltool.autogluon_adapter import AutoGluonAdapter, AutoGluonError, AutoGluonOutput
from mltool.cli import (
    features_project,
    leaderboard_project,
    plan_project,
    prepare_project,
    train_project,
    validate_project,
)
from mltool.config import (
    ConfigError,
    EvaluationConfig,
    ModelConfig,
    TaskConfig,
    TrainingConfig,
    load_config,
)
from mltool.evaluation import evaluate_predictions
from mltool.experiment import ExperimentError, build_experiment_plan, load_feature_artifacts
from mltool.training import (
    TrainingError,
    build_leaderboard,
    load_persisted_leaderboard,
    train_experiment,
)


def project_config(
    *,
    task: str = "binary",
    target: str = "label",
    positive_class: Any = 1,
    data_path: str = "./data/dataset.csv",
    features: dict[str, Any] | None = None,
    models: list[dict[str, Any]] | None = None,
    evaluation: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_config: dict[str, Any] = {"type": task, "target": target}
    if task == "binary" and positive_class is not None:
        task_config["positive_class"] = positive_class
    raw: dict[str, Any] = {
        "schema_version": "0.1",
        "project": {"name": "phase4-test"},
        "task": task_config,
        "data": {"format": "auto", "path": data_path},
        "validation": {"enabled": True, "fail_on_error": True},
        "split": {
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "stratify": "auto",
            "random_seed": 42,
        },
        "preprocessing": {
            "external": {"enabled": False, "entrypoint": None, "params": {}}
        },
        "features": features
        or {
            "plugins": [],
            "sets": [{"name": "base", "source_columns": ["*"], "plugins": []}],
        },
    }
    if models is not None:
        raw["models"] = models
    if evaluation is not None:
        raw["evaluation"] = evaluation
    if training is not None:
        raw["training"] = training
    return raw


def write_project(
    root: Path,
    *,
    frame: pd.DataFrame | None = None,
    raw: dict[str, Any] | None = None,
    parquet: bool = False,
) -> Path:
    (root / "data").mkdir(parents=True)
    frame = frame if frame is not None else pd.DataFrame(
        {"x": range(100), "z": [value % 7 for value in range(100)], "label": [value % 2 for value in range(100)]}
    )
    data_name = "dataset.parquet" if parquet else "dataset.csv"
    if parquet:
        frame.to_parquet(root / "data" / data_name, index=False)
    else:
        frame.to_csv(root / "data" / data_name, index=False)
    raw = raw or project_config(
        data_path=f"./data/{data_name}",
        models=[
            {"name": "lightgbm", "family": "GBM", "params": {}},
            {"name": "forest", "family": "RF", "params": {"num_boost_round": 3}},
        ],
    )
    raw["data"]["path"] = f"./data/{data_name}"
    path = root / "mltool.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def materialize(root: Path, **kwargs: Any) -> Path:
    config_path = write_project(root, **kwargs)
    assert prepare_project(config_path) == 0
    assert features_project(config_path) == 0
    return config_path


def test_models_config_parses_and_preserves_fixed_params(tmp_path: Path) -> None:
    raw = project_config(
        models=[{"name": "cat", "family": "CAT", "params": {"depth": 4}}]
    )
    config = load_config(write_project(tmp_path, raw=raw))
    assert config.models == [ModelConfig("cat", "CAT", {"depth": 4})]


@pytest.mark.parametrize(
    ("models", "message"),
    [
        (
            [
                {"name": "same", "family": "GBM", "params": {}},
                {"name": "same", "family": "RF", "params": {}},
            ],
            "duplicate model name",
        ),
        ([{"name": "../bad", "family": "GBM", "params": {}}], "unsafe"),
        ([{"name": "model", "family": "NN", "params": {}}], "unsupported model family"),
    ],
)
def test_invalid_model_identity_is_rejected(
    tmp_path: Path, models: list[dict[str, Any]], message: str
) -> None:
    path = write_project(tmp_path, raw=project_config(models=models))
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_model_params_must_be_json_serializable(tmp_path: Path) -> None:
    path = write_project(tmp_path, raw=project_config(models=[]))
    raw = yaml.safe_load(path.read_text())
    raw["models"] = [{"name": "bad", "family": "GBM", "params": {"x": float("nan")}}]
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON-serializable"):
        load_config(path)


def test_missing_models_remains_loadable_but_plan_rejects_it(tmp_path: Path) -> None:
    path = materialize(tmp_path, raw=project_config(models=None))
    config = load_config(path)
    assert config.models == []
    assert validate_project(path) == 0
    with pytest.raises(ExperimentError, match="no models are configured"):
        build_experiment_plan(config)


@pytest.mark.parametrize(
    ("task", "primary", "secondary"),
    [
        ("binary", "roc_auc", ["f1", "accuracy"]),
        ("multiclass", "accuracy", ["f1_macro"]),
        ("regression", "rmse", ["mae", "r2"]),
    ],
)
def test_default_evaluation_metrics_depend_on_task(
    tmp_path: Path, task: str, primary: str, secondary: list[str]
) -> None:
    frame = pd.DataFrame({"x": range(40), "label": [value % 3 for value in range(40)]})
    if task == "regression":
        frame["label"] = [value / 3 for value in range(40)]
    raw = project_config(task=task, positive_class=None, models=[])
    config = load_config(write_project(tmp_path, frame=frame, raw=raw))
    assert config.evaluation == EvaluationConfig(primary, secondary)


@pytest.mark.parametrize(
    ("task", "metric"),
    [("binary", "rmse"), ("multiclass", "roc_auc"), ("regression", "accuracy"), ("binary", "bogus")],
)
def test_task_incompatible_or_unknown_metric_is_rejected(
    tmp_path: Path, task: str, metric: str
) -> None:
    raw = project_config(
        task=task,
        positive_class=None,
        models=[],
        evaluation={"primary_metric": metric, "secondary_metrics": []},
    )
    with pytest.raises(ConfigError, match="not supported"):
        load_config(write_project(tmp_path, raw=raw))


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "10"])
def test_invalid_training_time_limit_is_rejected(tmp_path: Path, value: Any) -> None:
    raw = project_config(models=[], training={"time_limit_seconds": value})
    with pytest.raises(ConfigError, match="positive integer"):
        load_config(write_project(tmp_path, raw=raw))


def test_valid_training_time_limit_and_null(tmp_path: Path) -> None:
    first = load_config(
        write_project(tmp_path / "a", raw=project_config(models=[], training={"time_limit_seconds": 5}))
    )
    second = load_config(
        write_project(tmp_path / "b", raw=project_config(models=[], training={"time_limit_seconds": None}))
    )
    assert first.training == TrainingConfig(5)
    assert second.training == TrainingConfig(None)


def test_planner_builds_deterministic_cartesian_product_and_hashes(tmp_path: Path) -> None:
    features = {
        "plugins": [],
        "sets": [
            {"name": "base", "source_columns": ["*"], "plugins": []},
            {"name": "small", "source_columns": ["x"], "plugins": []},
        ],
    }
    raw = project_config(
        features=features,
        models=[
            {"name": "gbm", "family": "GBM", "params": {}},
            {"name": "rf", "family": "RF", "params": {}},
        ],
    )
    path = materialize(tmp_path, raw=raw)
    first = build_experiment_plan(load_config(path))
    second = build_experiment_plan(load_config(path))
    assert [candidate.candidate_id for candidate in first.candidates] == [
        "base__gbm",
        "base__rf",
        "small__gbm",
        "small__rf",
    ]
    assert [artifact.train_sha256 for artifact in first.feature_sets] == [
        artifact.train_sha256 for artifact in second.feature_sets
    ]
    assert [artifact.validation_sha256 for artifact in first.feature_sets] == [
        artifact.validation_sha256 for artifact in second.feature_sets
    ]


def test_plan_is_dry_run_and_creates_no_training_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = materialize(tmp_path)
    assert plan_project(path) == 0
    assert "Total candidates: 2" in capsys.readouterr().out
    assert not (tmp_path / ".mltool/training").exists()


def test_missing_feature_artifact_is_reported(tmp_path: Path) -> None:
    path = materialize(tmp_path)
    (tmp_path / ".mltool/features/base/train.parquet").unlink()
    with pytest.raises(ExperimentError, match="missing"):
        build_experiment_plan(load_config(path))


def test_changed_raw_source_fingerprint_is_detected(tmp_path: Path) -> None:
    path = materialize(tmp_path)
    with (tmp_path / "data/dataset.csv").open("a", encoding="utf-8") as stream:
        stream.write("100,2,0\n")
    # caught one phase earlier since Phase 12: prepare itself is stale
    with pytest.raises(ExperimentError, match="source dataset changed.*mltool prepare"):
        build_experiment_plan(load_config(path))


def test_changed_feature_set_recipe_is_detected(tmp_path: Path) -> None:
    path = materialize(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["features"]["sets"][0]["source_columns"] = ["x"]
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ExperimentError, match="source columns changed"):
        build_experiment_plan(load_config(path))


def test_changed_feature_plugin_config_is_detected(tmp_path: Path) -> None:
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "features.py").write_text(
        """import pandas as pd
class AddOne:
    def __init__(self, amount=1): self.amount = amount
    def fit(self, X): pass
    def transform(self, X):
        return pd.DataFrame({'added': X['x'] + self.amount}, index=X.index)
""",
        encoding="utf-8",
    )
    features = {
        "plugins": [
            {"name": "add", "entrypoint": "./plugins/features.py:AddOne", "params": {"amount": 1}}
        ],
        "sets": [{"name": "base", "source_columns": ["*"], "plugins": ["add"]}],
    }
    path = materialize(tmp_path, raw=project_config(features=features, models=[{"name": "gbm", "family": "GBM", "params": {}}]))
    raw = yaml.safe_load(path.read_text())
    raw["features"]["plugins"][0]["params"]["amount"] = 2
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ExperimentError, match="configuration changed"):
        load_feature_artifacts(load_config(path))


class FakePredictor:
    init_kwargs: dict[str, Any] = {}
    fit_kwargs: dict[str, Any] = {}
    validation_frames: list[pd.DataFrame] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).init_kwargs = kwargs
        self.positive_class = kwargs.get("positive_class", "yes")

    def fit(self, **kwargs: Any) -> "FakePredictor":
        type(self).fit_kwargs = kwargs
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        type(self).validation_frames.append(frame.copy())
        return pd.Series(["no", "yes"], index=frame.index)

    def predict_proba(self, frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        type(self).validation_frames.append(frame.copy())
        return pd.DataFrame({"no": [0.8, 0.1], "yes": [0.2, 0.9]}, index=frame.index)

    def model_names(self) -> list[str]:
        return ["LightGBM"]


def test_autogluon_adapter_isolated_fit_contract(tmp_path: Path) -> None:
    FakePredictor.validation_frames = []
    adapter = AutoGluonAdapter(
        predictor_factory=FakePredictor,
        version_resolver=lambda: "1.6.test",
    )
    train = pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]})
    validation = pd.DataFrame({"x": [4, 5]})
    output = adapter.fit_predict(
        train_data=train,
        validation_features=validation,
        task=TaskConfig("binary", "label", "yes"),
        model=ModelConfig("gbm", "GBM", {"num_boost_round": 7}),
        primary_metric="roc_auc",
        predictor_path=tmp_path / "predictor",
        training=TrainingConfig(9),
    )
    assert FakePredictor.init_kwargs["problem_type"] == "binary"
    assert FakePredictor.init_kwargs["eval_metric"] == "roc_auc"
    assert FakePredictor.init_kwargs["positive_class"] == "yes"
    fit = FakePredictor.fit_kwargs
    assert "tuning_data" not in fit
    assert fit["train_data"].equals(train)
    assert fit["hyperparameters"] == {"GBM": {"num_boost_round": 7}}
    assert fit["hyperparameter_tune_kwargs"] is None
    assert fit["fit_weighted_ensemble"] is False
    assert fit["fit_full_last_level_weighted_ensemble"] is False
    assert fit["num_bag_folds"] == 0
    assert fit["num_stack_levels"] == 0
    assert fit["num_gpus"] == 0
    assert fit["fit_strategy"] == "sequential"
    assert fit["time_limit"] == 9
    assert all(frame.equals(validation) for frame in FakePredictor.validation_frames)
    assert output.trained_models == ["LightGBM"]


def test_binary_metrics_use_non_one_positive_label_and_correct_probability_column() -> None:
    result = evaluate_predictions(
        task_type="binary",
        evaluation=EvaluationConfig("roc_auc", ["f1", "accuracy", "log_loss"]),
        y_true=pd.Series(["no", "yes", "no", "yes"]),
        predictions=pd.Series(["no", "yes", "no", "no"]),
        probabilities=pd.DataFrame(
            {"yes": [0.1, 0.9, 0.2, 0.4], "no": [0.9, 0.1, 0.8, 0.6]}
        ),
        positive_class="yes",
    )
    assert result["roc_auc"] == 1.0
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["accuracy"] == 0.75
    assert result["log_loss"] > 0


def test_multiclass_metrics_align_named_probability_columns() -> None:
    result = evaluate_predictions(
        task_type="multiclass",
        evaluation=EvaluationConfig("accuracy", ["f1_macro", "log_loss"]),
        y_true=pd.Series(["z", "a", "m"]),
        predictions=pd.Series(["z", "a", "m"]),
        probabilities=pd.DataFrame(
            {"a": [0.05, 0.9, 0.05], "m": [0.05, 0.05, 0.9], "z": [0.9, 0.05, 0.05]}
        ),
    )
    assert result["accuracy"] == 1.0
    assert result["f1_macro"] == 1.0
    assert result["log_loss"] < 0.2


def test_regression_metrics() -> None:
    result = evaluate_predictions(
        task_type="regression",
        evaluation=EvaluationConfig("rmse", ["mae", "r2"]),
        y_true=pd.Series([1.0, 2.0, 3.0]),
        predictions=pd.Series([1.0, 2.0, 4.0]),
    )
    assert result["rmse"] == pytest.approx((1 / 3) ** 0.5)
    assert result["mae"] == pytest.approx(1 / 3)
    assert result["r2"] == 0.5


class RecordingAdapter:
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
        token = {"GBM": "LightGBM", "RF": "RandomForest"}[model.family]
        return AutoGluonOutput(
            predictions=predictions,
            probabilities=probabilities,
            positive_class=positive,
            trained_models=[token],
            autogluon_version="1.6.fake",
        )


def test_training_persists_results_manifest_leaderboard_and_never_reads_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = materialize(tmp_path)
    plan = build_experiment_plan(load_config(path))
    RecordingAdapter.calls = []
    RecordingAdapter.failures = set()
    import mltool.training as training_module

    real_read = training_module.pd.read_parquet

    def guarded_read(path_arg: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        assert Path(path_arg).name != "test.parquet"
        return real_read(path_arg, *args, **kwargs)

    monkeypatch.setattr(training_module.pd, "read_parquet", guarded_read)
    result = train_experiment(plan, adapter_factory=RecordingAdapter)
    assert result.successful_count == 2
    assert len(RecordingAdapter.calls) == 2
    assert all("label" in call["train_data"] for call in RecordingAdapter.calls)
    assert all("label" not in call["validation_features"] for call in RecordingAdapter.calls)
    training_path = tmp_path / ".mltool/training"
    assert (training_path / "manifest.json").is_file()
    assert (training_path / "leaderboard.csv").is_file()
    assert (training_path / "leaderboard.json").is_file()
    for candidate in ("base__lightgbm", "base__forest"):
        result_json = json.loads(
            (training_path / "candidates" / candidate / "result.json").read_text()
        )
        assert result_json["status"] == "SUCCEEDED"
        assert result_json["metrics"]["roc_auc"] == 1.0
    manifest = json.loads((training_path / "manifest.json").read_text())
    assert manifest["test_data_used"] is False
    assert manifest["successful_count"] == 2


def test_one_failed_candidate_does_not_stop_later_candidates(tmp_path: Path) -> None:
    path = materialize(tmp_path)
    plan = build_experiment_plan(load_config(path))
    RecordingAdapter.calls = []
    RecordingAdapter.failures = {"lightgbm"}
    result = train_experiment(plan, adapter_factory=RecordingAdapter)
    assert [item["status"] for item in result.candidate_results] == ["FAILED", "SUCCEEDED"]
    assert result.is_successful
    failed = json.loads(
        (tmp_path / ".mltool/training/candidates/base__lightgbm/result.json").read_text()
    )
    assert failed["status"] == "FAILED"
    assert "intentional failure" in failed["error_message"]


def test_zero_successful_candidates_is_an_overall_failure_result(tmp_path: Path) -> None:
    path = materialize(tmp_path)
    plan = build_experiment_plan(load_config(path))
    RecordingAdapter.failures = {"lightgbm", "forest"}
    result = train_experiment(plan, adapter_factory=RecordingAdapter)
    assert not result.is_successful
    assert result.failed_count == 2
    assert all(row["status"] == "FAILED" for row in result.leaderboard)


def test_regression_numeric_string_target_is_converted_only_in_candidate_copy(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"x": range(100), "label": [f"{value}.0" for value in range(100)]})
    raw = project_config(
        task="regression",
        positive_class=None,
        data_path="./data/dataset.parquet",
        models=[{"name": "gbm", "family": "GBM", "params": {}}],
    )
    path = materialize(tmp_path, frame=frame, raw=raw, parquet=True)
    feature_bytes = (tmp_path / ".mltool/features/base/train.parquet").read_bytes()
    RecordingAdapter.calls = []
    RecordingAdapter.failures = set()
    result = train_experiment(
        build_experiment_plan(load_config(path)), adapter_factory=RecordingAdapter
    )
    assert result.is_successful
    assert result.candidate_results[0]["target_conversion_applied"] is True
    assert is_numeric_dtype(RecordingAdapter.calls[0]["train_data"]["label"])
    assert (tmp_path / ".mltool/features/base/train.parquet").read_bytes() == feature_bytes


def test_classification_target_is_not_coerced(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"x": range(100), "label": ["yes" if value % 2 else "no" for value in range(100)]}
    )
    raw = project_config(
        positive_class="yes",
        data_path="./data/dataset.parquet",
        models=[{"name": "gbm", "family": "GBM", "params": {}}],
    )
    path = materialize(tmp_path, frame=frame, raw=raw, parquet=True)
    RecordingAdapter.calls = []
    RecordingAdapter.failures = set()
    result = train_experiment(
        build_experiment_plan(load_config(path)), adapter_factory=RecordingAdapter
    )
    assert result.is_successful
    assert result.candidate_results[0]["target_conversion_applied"] is False
    assert RecordingAdapter.calls[0]["train_data"]["label"].dtype == object


def test_leaderboard_sorting_respects_direction_and_failures_rank_last() -> None:
    results = [
        {"candidate_id": "a", "feature_set": "f", "model": {"name": "a", "family": "GBM"}, "metrics": {"rmse": 2.0}, "training_seconds": 1.0, "status": "SUCCEEDED"},
        {"candidate_id": "b", "feature_set": "f", "model": {"name": "b", "family": "RF"}, "metrics": {"rmse": 1.0}, "training_seconds": 1.0, "status": "SUCCEEDED"},
        {"candidate_id": "c", "feature_set": "f", "model": {"name": "c", "family": "RF"}, "status": "FAILED", "error_message": "x"},
    ]
    minimized = build_leaderboard(results, primary_metric="rmse", direction="minimize")
    maximized_results = [replace_result | {"metrics": {"roc_auc": score}} for replace_result, score in zip(results[:2], [0.8, 0.9], strict=True)] + [results[2]]
    maximized = build_leaderboard(maximized_results, primary_metric="roc_auc", direction="maximize")
    assert [row["candidate_id"] for row in minimized] == ["b", "a", "c"]
    assert [row["candidate_id"] for row in maximized] == ["b", "a", "c"]
    assert minimized[-1]["rank"] is None


def test_persisted_leaderboard_is_read_only_and_warns_on_config_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = materialize(tmp_path)
    RecordingAdapter.failures = set()
    train_experiment(build_experiment_plan(load_config(path)), adapter_factory=RecordingAdapter)
    before = (tmp_path / ".mltool/training/leaderboard.json").read_bytes()
    import mltool.cli as cli_module

    monkeypatch.setattr(
        cli_module,
        "train_experiment",
        lambda *args, **kwargs: pytest.fail("leaderboard must not retrain"),
    )
    assert leaderboard_project(path) == 0
    assert "MLTool global leaderboard" in capsys.readouterr().out
    assert (tmp_path / ".mltool/training/leaderboard.json").read_bytes() == before

    raw = yaml.safe_load(path.read_text())
    raw["models"] = [{"name": "only", "family": "GBM", "params": {}}]
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    persisted = load_persisted_leaderboard(load_config(path))
    assert persisted.warning is not None


def test_leaderboard_requires_training_artifacts(tmp_path: Path) -> None:
    path = write_project(tmp_path)
    with pytest.raises(TrainingError, match='run "mltool train"'):
        load_persisted_leaderboard(load_config(path))


def test_plan_train_and_leaderboard_expected_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    no_models = materialize(tmp_path / "no-models", raw=project_config(models=None))
    assert plan_project(no_models) == 2

    path = materialize(tmp_path / "with-models")
    fake_success = SimpleNamespace(is_successful=True, render=lambda: "trained")
    monkeypatch.setattr("mltool.cli.train_experiment", lambda plan: fake_success)
    assert train_project(path) == 0
    fake_failure = SimpleNamespace(is_successful=False, render=lambda: "failed")
    monkeypatch.setattr("mltool.cli.train_experiment", lambda plan: fake_failure)
    assert train_project(path) == 1
    assert leaderboard_project(path) == 2


def test_phase1_through_phase3_commands_remain_working(tmp_path: Path) -> None:
    path = write_project(tmp_path)
    assert validate_project(path) == 0
    assert prepare_project(path) == 0
    assert features_project(path) == 0
