from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cloudpickle
import pandas as pd
import pytest
import yaml

import mltool.cli as cli
from mltool.cli import init_project, prepare_project
from mltool.config import ConfigError, SplitConfig, load_config
from mltool.data import load_dataset
from mltool.preparation import prepare_dataset
from mltool.splitting import SplitError, split_dataset
from mltool.validation import validate_dataset


def binary_frame(rows: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": range(rows),
            "value": [float(index * 2) for index in range(rows)],
            "drop_me": [f"group-{index % 4}" for index in range(rows)],
            "label": [index % 2 for index in range(rows)],
        }
    )


def write_config(
    root: Path,
    *,
    task_type: str = "binary",
    target: str = "label",
    data_path: str = "data/dataset.csv",
    data_format: str = "auto",
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: str | bool = "auto",
    random_seed: int = 42,
    preprocessing_enabled: bool = False,
    entrypoint: str | None = None,
    params: dict[str, object] | None = None,
) -> Path:
    task: dict[str, object] = {"type": task_type, "target": target}
    if task_type == "binary":
        task["positive_class"] = 1
    config = {
        "schema_version": "0.1",
        "project": {"name": "phase-2-test"},
        "task": task,
        "data": {"format": data_format, "path": data_path},
        "validation": {"enabled": True, "fail_on_error": True},
        "split": {
            "validation_ratio": validation_ratio,
            "test_ratio": test_ratio,
            "stratify": stratify,
            "random_seed": random_seed,
        },
        "preprocessing": {
            "external": {
                "enabled": preprocessing_enabled,
                "entrypoint": entrypoint,
                "params": params or {},
            }
        },
    }
    path = root / "mltool.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def write_csv_project(root: Path, frame: pd.DataFrame, **config_kwargs: object) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    frame.to_csv(data_dir / "dataset.csv", index=False)
    return write_config(root, **config_kwargs)


def load_valid_project(config_path: Path):
    config = load_config(config_path)
    dataset = load_dataset(config.data)
    report = validate_dataset(config, dataset)
    assert report.is_valid, report.errors
    return config, dataset, report


def test_binary_stratified_split_is_deterministic_and_approximately_70_15_15() -> None:
    frame = binary_frame(100)
    config = SplitConfig(
        validation_ratio=0.15,
        test_ratio=0.15,
        stratify="auto",
        random_seed=42,
    )

    first = split_dataset(frame, target="label", task_type="binary", config=config)
    second = split_dataset(frame, target="label", task_type="binary", config=config)

    assert first.stratified is True
    assert (len(first.train), len(first.validation), len(first.test)) == (70, 15, 15)
    assert first.train.index.tolist() == second.train.index.tolist()
    assert first.validation.index.tolist() == second.validation.index.tolist()
    assert first.test.index.tolist() == second.test.index.tolist()
    for split in (first.train, first.validation, first.test):
        assert set(split["label"]) == {0, 1}


def test_multiclass_auto_split_is_stratified() -> None:
    frame = pd.DataFrame(
        {"row_id": range(120), "label": [index % 3 for index in range(120)]}
    )

    result = split_dataset(
        frame,
        target="label",
        task_type="multiclass",
        config=SplitConfig(),
    )

    assert result.stratified is True
    for split in (result.train, result.validation, result.test):
        assert set(split["label"]) == {0, 1, 2}


def test_regression_auto_split_is_not_stratified() -> None:
    frame = pd.DataFrame(
        {"row_id": range(40), "price": [index / 3 for index in range(40)]}
    )

    result = split_dataset(
        frame,
        target="price",
        task_type="regression",
        config=SplitConfig(),
    )

    assert result.stratified is False
    assert sum(map(len, (result.train, result.validation, result.test))) == 40


def test_explicit_false_disables_classification_stratification() -> None:
    result = split_dataset(
        binary_frame(40),
        target="label",
        task_type="binary",
        config=SplitConfig(stratify=False),
    )

    assert result.stratified is False


def test_impossible_stratification_fails_clearly() -> None:
    frame = binary_frame(6)

    with pytest.raises(SplitError, match="stratified split is impossible"):
        split_dataset(
            frame,
            target="label",
            task_type="binary",
            config=SplitConfig(),
        )


@pytest.mark.parametrize(
    ("validation_ratio", "test_ratio", "message"),
    [
        (0, 0.15, "validation_ratio"),
        (1, 0.15, "validation_ratio"),
        (0.15, 0, "test_ratio"),
        (0.15, 1, "test_ratio"),
        (0.5, 0.5, "must be less than 1"),
    ],
)
def test_invalid_split_ratios_are_rejected(
    tmp_path: Path,
    validation_ratio: float,
    test_ratio: float,
    message: str,
) -> None:
    config_path = write_config(
        tmp_path,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    with pytest.raises(ConfigError, match=message):
        load_config(config_path)


@pytest.mark.parametrize("random_seed", [True, 1.5, "42"])
def test_random_seed_must_be_an_integer(tmp_path: Path, random_seed: object) -> None:
    config_path = write_config(tmp_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["split"]["random_seed"] = random_seed
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="random_seed"):
        load_config(config_path)


def test_invalid_phase1_dataset_never_reaches_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_csv_project(
        tmp_path, pd.DataFrame({"feature": range(20)})
    )
    called = False

    def should_not_prepare(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("invalid data reached preparation")

    monkeypatch.setattr(cli, "prepare_dataset", should_not_prepare)

    assert prepare_project(config_path) == 2
    assert called is False
    assert 'target column "label" was not found' in capsys.readouterr().out
    assert not (tmp_path / ".mltool").exists()


def test_preprocessing_disabled_writes_splits_manifest_and_preserves_source(
    tmp_path: Path,
) -> None:
    config_path = write_csv_project(tmp_path, binary_frame())
    source_path = tmp_path / "data/dataset.csv"
    source_before = source_path.read_bytes()
    source_hash = hashlib.sha256(source_before).hexdigest()
    config, dataset, report = load_valid_project(config_path)

    result = prepare_dataset(config, dataset, warning_messages=report.warnings)

    prepared_dir = tmp_path / ".mltool/prepared"
    assert result.preprocessor_path is None
    assert not (prepared_dir / "preprocessor.pkl").exists()
    for name in ("train.parquet", "validation.parquet", "test.parquet"):
        assert (prepared_dir / name).is_file()
    assert (prepared_dir / "manifest.json").is_file()
    assert source_path.read_bytes() == source_before
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == {
        "path": str(source_path.resolve()),
        "format": "csv",
        "fingerprint": source_hash,
    }
    assert manifest["task"] == {"type": "binary", "target": "label"}
    assert manifest["split"]["stratified"] is True
    assert manifest["split"]["train_rows"] == 70
    assert manifest["split"]["validation_rows"] == 15
    assert manifest["split"]["test_rows"] == 15
    assert manifest["preprocessing"] == {"enabled": False}
    expected = split_dataset(
        dataset.frame,
        target=config.task.target,
        task_type=config.task.type,
        config=config.split,
    )
    for name in ("train", "validation", "test"):
        actual_frame = pd.read_parquet(prepared_dir / f"{name}.parquet")
        expected_frame = getattr(expected, name).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual_frame, expected_frame)


def test_repeated_preparation_same_seed_produces_equivalent_splits(tmp_path: Path) -> None:
    config_path = write_csv_project(tmp_path, binary_frame())
    config, dataset, report = load_valid_project(config_path)

    prepare_dataset(config, dataset, warning_messages=report.warnings)
    first = {
        name: pd.read_parquet(tmp_path / f".mltool/prepared/{name}.parquet")
        for name in ("train", "validation", "test")
    }
    prepare_dataset(config, dataset, warning_messages=report.warnings)
    second = {
        name: pd.read_parquet(tmp_path / f".mltool/prepared/{name}.parquet")
        for name in ("train", "validation", "test")
    }

    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


AUDIT_PLUGIN = '''
class AuditPreprocessor:
    def __init__(self, multiplier=1, drop_column="drop_me"):
        self.multiplier = multiplier
        self.drop_column = drop_column
        self.fit_calls = 0
        self.fit_columns = None
        self.fit_row_ids = None
        self.mean_ = None
        self.transform_row_ids = []

    def fit(self, X_train):
        self.fit_calls += 1
        self.fit_columns = list(X_train.columns)
        self.fit_row_ids = X_train["row_id"].tolist()
        self.mean_ = X_train["value"].mean()

    def transform(self, X):
        self.transform_row_ids.append(X["row_id"].tolist())
        result = X.copy()
        result["centered"] = (result["value"] - self.mean_) * self.multiplier
        result["renamed_value"] = result.pop("value")
        return result.drop(columns=[self.drop_column])
'''


def test_external_plugin_is_train_only_stateful_aligned_and_persisted(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "preprocess.py").write_text(AUDIT_PLUGIN, encoding="utf-8")
    config_path = write_csv_project(
        tmp_path,
        binary_frame(),
        preprocessing_enabled=True,
        entrypoint="./plugins/preprocess.py:AuditPreprocessor",
        params={"multiplier": 3, "drop_column": "drop_me"},
    )
    config, dataset, report = load_valid_project(config_path)

    result = prepare_dataset(config, dataset, warning_messages=report.warnings)

    assert result.preprocessor_path is not None
    assert result.preprocessor_path.is_file()
    with result.preprocessor_path.open("rb") as stream:
        fitted = cloudpickle.load(stream)
    assert fitted.fit_calls == 1
    assert "label" not in fitted.fit_columns
    assert set(fitted.fit_columns) == {"row_id", "value", "drop_me"}
    assert len(fitted.fit_row_ids) == 70
    assert [len(values) for values in fitted.transform_row_ids] == [70, 15, 15]
    assert fitted.transform_row_ids[0] == fitted.fit_row_ids
    all_transformed = [value for values in fitted.transform_row_ids for value in values]
    assert sorted(all_transformed) == list(range(100))

    for split_name in ("train", "validation", "test"):
        prepared = pd.read_parquet(tmp_path / f".mltool/prepared/{split_name}.parquet")
        assert list(prepared.columns) == [
            "row_id",
            "centered",
            "renamed_value",
            "label",
        ]
        assert prepared["label"].tolist() == [
            row_id % 2 for row_id in prepared["row_id"]
        ]
        expected_centered = (prepared["renamed_value"] - fitted.mean_) * 3
        pd.testing.assert_series_equal(
            prepared["centered"], expected_centered, check_names=False
        )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["preprocessing"]["enabled"] is True
    assert manifest["preprocessing"]["entrypoint"] == (
        "./plugins/preprocess.py:AuditPreprocessor"
    )
    assert manifest["preprocessing"]["params"] == {
        "drop_column": "drop_me",
        "multiplier": 3,
    }
    assert manifest["preprocessing"]["artifact"] == str(result.preprocessor_path)


@pytest.mark.parametrize(
    ("transform_body", "expected_error"),
    [
        ("return X.iloc[:-1].copy()", "changed row count"),
        ("return X.iloc[::-1].copy()", "changed or reordered the index"),
        ("return X.to_dict()", "must return a pandas DataFrame"),
        (
            'result = X.copy()\n        result.columns = ["duplicate"] * len(result.columns)\n        return result',
            "produced duplicate columns",
        ),
        (
            'result = X.copy()\n        result["label"] = 0\n        return result',
            "produced configured target column",
        ),
    ],
)
def test_transform_contract_violations_fail_without_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    transform_body: str,
    expected_error: str,
) -> None:
    plugin = tmp_path / "plugin.py"
    plugin.write_text(
        "class BadPreprocessor:\n"
        "    def fit(self, X_train):\n"
        "        pass\n"
        "    def transform(self, X):\n"
        f"        {transform_body}\n",
        encoding="utf-8",
    )
    config_path = write_csv_project(
        tmp_path,
        binary_frame(40),
        preprocessing_enabled=True,
        entrypoint="./plugin.py:BadPreprocessor",
    )

    assert prepare_project(config_path) == 2
    output = capsys.readouterr().out
    assert expected_error in output
    assert "Result: FAILED" in output
    assert "Traceback" not in output
    assert not (tmp_path / ".mltool/prepared").exists()


def test_missing_plugin_file_fails_clearly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_csv_project(
        tmp_path,
        binary_frame(40),
        preprocessing_enabled=True,
        entrypoint="./missing.py:Missing",
    )

    assert prepare_project(config_path) == 2
    assert "external preprocessor file not found" in capsys.readouterr().out


def test_missing_plugin_class_fails_clearly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "plugin.py").write_text("class Existing: pass\n", encoding="utf-8")
    config_path = write_csv_project(
        tmp_path,
        binary_frame(40),
        preprocessing_enabled=True,
        entrypoint="./plugin.py:Missing",
    )

    assert prepare_project(config_path) == 2
    assert 'class "Missing" was not found' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("class_body", "expected_error"),
    [
        (
            "class Incomplete:\n"
            "    def transform(self, X):\n"
            "        return X\n",
            "must provide callable fit",
        ),
        (
            "class Incomplete:\n"
            "    def fit(self, X):\n"
            "        pass\n",
            "must provide callable transform",
        ),
    ],
)
def test_plugin_without_required_method_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    class_body: str,
    expected_error: str,
) -> None:
    (tmp_path / "plugin.py").write_text(class_body, encoding="utf-8")
    config_path = write_csv_project(
        tmp_path,
        binary_frame(40),
        preprocessing_enabled=True,
        entrypoint="./plugin.py:Incomplete",
    )

    assert prepare_project(config_path) == 2
    assert expected_error in capsys.readouterr().out


def test_plugin_module_and_constructor_failures_are_expected_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plugin = tmp_path / "plugin.py"
    plugin.write_text("this is not valid python !\n", encoding="utf-8")
    config_path = write_csv_project(
        tmp_path,
        binary_frame(40),
        preprocessing_enabled=True,
        entrypoint="./plugin.py:Broken",
    )
    assert prepare_project(config_path) == 2
    assert "could not load external preprocessor module" in capsys.readouterr().out

    plugin.write_text(
        "class Broken:\n"
        "    def __init__(self, required): pass\n"
        "    def fit(self, X): pass\n"
        "    def transform(self, X): return X\n",
        encoding="utf-8",
    )
    assert prepare_project(config_path) == 2
    assert "could not instantiate external preprocessor" in capsys.readouterr().out


def test_plugin_system_exit_is_an_expected_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "plugin.py").write_text("raise SystemExit('plugin stopped')\n", encoding="utf-8")
    config_path = write_csv_project(
        tmp_path,
        binary_frame(40),
        preprocessing_enabled=True,
        entrypoint="./plugin.py:Stopped",
    )

    assert prepare_project(config_path) == 2
    output = capsys.readouterr().out
    assert "could not load external preprocessor module" in output
    assert "plugin stopped" in output
    assert "Traceback" not in output


def test_preprocessor_serialization_failure_is_clear_and_not_materialized(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "plugin.py").write_text(
        "class Unserializable:\n"
        "    def fit(self, X):\n"
        "        self.generator = (value for value in X.iloc[:, 0])\n"
        "    def transform(self, X):\n"
        "        return X.copy()\n",
        encoding="utf-8",
    )
    config_path = write_csv_project(
        tmp_path,
        binary_frame(40),
        preprocessing_enabled=True,
        entrypoint="./plugin.py:Unserializable",
    )

    assert prepare_project(config_path) == 2
    output = capsys.readouterr().out
    assert "could not serialize fitted external preprocessor" in output
    assert "Result: FAILED" in output
    assert not (tmp_path / ".mltool/prepared").exists()


def test_regression_numeric_string_target_is_preserved_exactly(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = pd.DataFrame(
        {
            "row_id": range(40),
            "price": [str(index + 0.25) for index in range(40)],
        }
    )
    source.to_parquet(data_dir / "dataset.parquet", index=False)
    config_path = write_config(
        tmp_path,
        task_type="regression",
        target="price",
        data_path="data/dataset.parquet",
    )
    config, dataset, report = load_valid_project(config_path)

    prepare_dataset(config, dataset, warning_messages=report.warnings)

    prepared_values: list[str] = []
    for name in ("train", "validation", "test"):
        prepared_values.extend(
            pd.read_parquet(tmp_path / f".mltool/prepared/{name}.parquet")[
                "price"
            ].tolist()
        )
    assert sorted(prepared_values) == sorted(source["price"].tolist())
    assert all(isinstance(value, str) for value in prepared_values)


def test_prepare_cli_success_and_expected_error_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    valid_root = tmp_path / "valid"
    invalid_root = tmp_path / "invalid"
    valid_root.mkdir()
    invalid_root.mkdir()
    valid_config = write_csv_project(valid_root, binary_frame(40))
    invalid_config = write_csv_project(
        invalid_root, pd.DataFrame({"feature": range(40)})
    )

    assert prepare_project(valid_config) == 0
    valid_output = capsys.readouterr().out
    assert "Result: PREPARED" in valid_output
    assert "Strategy: stratified" in valid_output

    assert prepare_project(invalid_config) == 2
    invalid_output = capsys.readouterr().out
    assert "Result: INVALID" in invalid_output
    assert "Traceback" not in invalid_output


def test_valid_dataset_warnings_are_shown_and_do_not_stop_preparation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = binary_frame(40)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    config_path = write_csv_project(tmp_path, frame)

    assert prepare_project(config_path) == 0
    output = capsys.readouterr().out
    assert "VALID with 1 warning" in output
    assert "1 duplicate row(s) found" in output
    assert "Result: PREPARED" in output


def test_prepare_validates_even_when_validation_is_configured_disabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_csv_project(
        tmp_path, pd.DataFrame({"feature": range(40)})
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["validation"]["enabled"] = False
    raw["validation"]["fail_on_error"] = False
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    assert prepare_project(config_path) == 2
    output = capsys.readouterr().out
    assert 'target column "label" was not found' in output
    assert "Result: INVALID" in output
    assert not (tmp_path / ".mltool").exists()


def test_init_template_contains_only_supported_phase_sections(tmp_path: Path) -> None:
    assert init_project(tmp_path) == 0

    raw = yaml.safe_load((tmp_path / "mltool.yaml").read_text(encoding="utf-8"))
    assert set(raw) == {
        "schema_version",
        "project",
        "task",
        "data",
        "validation",
        "split",
        "preprocessing",
        "features",
    }
    assert raw["split"] == {
        "validation_ratio": 0.15,
        "test_ratio": 0.15,
        "stratify": "auto",
        "random_seed": 42,
    }
    assert raw["preprocessing"] == {
        "external": {"enabled": False, "entrypoint": None, "params": {}}
    }
    assert raw["features"] == {
        "plugins": [],
        "sets": [{"name": "base", "source_columns": ["*"], "plugins": []}],
    }
