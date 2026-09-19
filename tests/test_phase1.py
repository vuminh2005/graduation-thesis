from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mltool.cli import init_project, validate_project
from mltool.config import ConfigError, load_config
from mltool.data import load_dataset
from mltool.validation import validate_dataset


def write_config(
    root: Path,
    *,
    task_type: str = "binary",
    target: str = "label",
    data_path: str = "data/dataset.csv",
    data_format: str = "auto",
    positive_class: object = 1,
) -> Path:
    task: dict[str, object] = {"type": task_type, "target": target}
    if task_type == "binary":
        task["positive_class"] = positive_class
    config = {
        "schema_version": "0.1",
        "project": {"name": "test-project"},
        "task": task,
        "data": {"format": data_format, "path": data_path},
        "validation": {"enabled": True, "fail_on_error": True},
    }
    path = root / "mltool.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def report_for(root: Path):
    config = load_config(root / "mltool.yaml")
    return validate_dataset(config, load_dataset(config.data))


def test_valid_binary_csv(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    dataset_path = data / "dataset.csv"
    pd.DataFrame(
        {"age": [20, 30, 40, 50], "label": [0, 1, 0, 1]}
    ).to_csv(dataset_path, index=False)
    write_config(tmp_path)

    report = report_for(tmp_path)

    assert report.is_valid
    assert report.dataset is not None
    assert report.dataset.rows == 4
    assert report.dataset.columns == 2
    assert report.dataset.fingerprint == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert report.target_distribution == [(0, 2), (1, 2)]


def test_valid_multiclass_dataset(tmp_path: Path) -> None:
    pd.DataFrame(
        {"feature": range(6), "species": ["a", "b", "c", "a", "b", "c"]}
    ).to_csv(tmp_path / "dataset.csv", index=False)
    write_config(
        tmp_path,
        task_type="multiclass",
        target="species",
        data_path="dataset.csv",
    )

    report = report_for(tmp_path)

    assert report.is_valid
    assert report.target_kind == "3"


def test_valid_regression_dataset(tmp_path: Path) -> None:
    pd.DataFrame(
        {"feature": [1, 2, 3], "price": [1.5, 2.75, 4.0]}
    ).to_csv(tmp_path / "dataset.csv", index=False)
    write_config(
        tmp_path,
        task_type="regression",
        target="price",
        data_path="dataset.csv",
    )

    assert report_for(tmp_path).is_valid


def test_missing_dataset_file_is_invalid(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_config(tmp_path)

    assert validate_project(config_path) == 2
    output = capsys.readouterr().out
    assert "dataset path does not exist" in output
    assert "Result: INVALID" in output


def test_missing_target_column(tmp_path: Path) -> None:
    pd.DataFrame({"feature": [1, 2]}).to_csv(tmp_path / "dataset.csv", index=False)
    write_config(tmp_path, data_path="dataset.csv")

    report = report_for(tmp_path)

    assert not report.is_valid
    assert 'target column "label" was not found' in report.errors


def test_target_containing_null_values(tmp_path: Path) -> None:
    pd.DataFrame(
        {"feature": [1, 2, 3], "label": [0, None, 1]}
    ).to_csv(tmp_path / "dataset.csv", index=False)
    write_config(tmp_path, data_path="dataset.csv")

    report = report_for(tmp_path)

    assert not report.is_valid
    assert any("contains 1 null value" in error for error in report.errors)


def test_entirely_null_target_is_invalid(tmp_path: Path) -> None:
    pd.DataFrame({"feature": [1, 2], "label": [None, None]}).to_csv(
        tmp_path / "dataset.csv", index=False
    )
    write_config(tmp_path, data_path="dataset.csv")

    report = report_for(tmp_path)

    assert not report.is_valid
    assert 'target column "label" is entirely null' in report.errors


@pytest.mark.parametrize("labels", [[1, 1, 1], [0, 1, 2]])
def test_binary_target_requires_exactly_two_classes(tmp_path: Path, labels: list[int]) -> None:
    pd.DataFrame({"feature": range(len(labels)), "label": labels}).to_csv(
        tmp_path / "dataset.csv", index=False
    )
    write_config(tmp_path, data_path="dataset.csv")

    report = report_for(tmp_path)

    assert not report.is_valid
    assert any("exactly 2" in error for error in report.errors)


def test_empty_dataset(tmp_path: Path) -> None:
    pd.DataFrame(columns=["feature", "label"]).to_csv(
        tmp_path / "dataset.csv", index=False
    )
    write_config(tmp_path, data_path="dataset.csv")

    report = report_for(tmp_path)

    assert not report.is_valid
    assert "dataset has zero rows" in report.errors


def test_completely_empty_file_has_zero_rows_and_columns(tmp_path: Path) -> None:
    (tmp_path / "dataset.csv").write_bytes(b"")
    write_config(tmp_path, data_path="dataset.csv")

    report = report_for(tmp_path)

    assert "dataset has zero rows" in report.errors
    assert "dataset has zero columns" in report.errors


def test_warns_for_duplicate_rows(tmp_path: Path) -> None:
    pd.DataFrame({"feature": [1, 1, 2], "label": [0, 0, 1]}).to_csv(
        tmp_path / "dataset.csv", index=False
    )
    write_config(tmp_path, data_path="dataset.csv")

    report = report_for(tmp_path)

    assert report.is_valid
    assert any("1 duplicate row" in warning for warning in report.warnings)


def test_warns_for_missing_feature_values(tmp_path: Path) -> None:
    pd.DataFrame({"feature": [1, None, 3], "label": [0, 1, 0]}).to_csv(
        tmp_path / "dataset.csv", index=False
    )
    write_config(tmp_path, data_path="dataset.csv")

    report = report_for(tmp_path)

    assert report.is_valid
    assert any(
        'feature "feature" contains 1 missing value' in warning
        for warning in report.warnings
    )


def test_warns_for_constant_feature_columns(tmp_path: Path) -> None:
    pd.DataFrame({"constant": [7, 7, 7], "label": [0, 1, 0]}).to_csv(
        tmp_path / "dataset.csv", index=False
    )
    write_config(tmp_path, data_path="dataset.csv")

    report = report_for(tmp_path)

    assert report.is_valid
    assert 'feature "constant" is constant' in report.warnings


def test_parquet_loading(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    pd.DataFrame({"feature": [1, 2, 3], "label": [0, 1, 0]}).to_parquet(
        dataset_path, index=False
    )
    write_config(tmp_path, data_path="dataset.parquet")

    report = report_for(tmp_path)

    assert report.is_valid
    assert report.dataset is not None
    assert report.dataset.format == "parquet"
    assert report.dataset.fingerprint == hashlib.sha256(dataset_path.read_bytes()).hexdigest()


def test_init_creates_expected_files(tmp_path: Path) -> None:
    assert init_project(tmp_path) == 0

    assert (tmp_path / "mltool.yaml").is_file()
    assert (tmp_path / "data").is_dir()
    config = load_config(tmp_path / "mltool.yaml")
    assert config.schema_version == "0.1"
    assert config.task.type == "binary"
    assert config.data.path == (tmp_path / "data/dataset.csv").resolve()


def test_init_refuses_to_overwrite_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "mltool.yaml"
    config_path.write_text("sentinel", encoding="utf-8")

    assert init_project(tmp_path) == 2

    assert config_path.read_text(encoding="utf-8") == "sentinel"
    assert "refusing to overwrite" in capsys.readouterr().err


def test_duplicate_csv_column_names_are_invalid(tmp_path: Path) -> None:
    (tmp_path / "dataset.csv").write_text(
        "feature,feature,label\n1,2,0\n3,4,1\n", encoding="utf-8"
    )
    write_config(tmp_path, data_path="dataset.csv")

    report = report_for(tmp_path)

    assert not report.is_valid
    assert 'duplicate column names found: "feature"' in report.errors


def test_regression_rejects_non_numeric_target(tmp_path: Path) -> None:
    pd.DataFrame({"feature": [1, 2], "price": ["low", "high"]}).to_csv(
        tmp_path / "dataset.csv", index=False
    )
    write_config(
        tmp_path,
        task_type="regression",
        target="price",
        data_path="dataset.csv",
    )

    report = report_for(tmp_path)

    assert not report.is_valid
    assert any("not usable as numeric" in error for error in report.errors)


def test_multiclass_requires_at_least_two_classes(tmp_path: Path) -> None:
    pd.DataFrame({"feature": [1, 2], "label": ["only", "only"]}).to_csv(
        tmp_path / "dataset.csv", index=False
    )
    write_config(
        tmp_path,
        task_type="multiclass",
        target="label",
        data_path="dataset.csv",
    )

    report = report_for(tmp_path)

    assert not report.is_valid
    assert any("at least 2" in error for error in report.errors)


def test_binary_positive_class_must_exist(tmp_path: Path) -> None:
    pd.DataFrame({"feature": [1, 2], "label": [0, 1]}).to_csv(
        tmp_path / "dataset.csv", index=False
    )
    write_config(tmp_path, data_path="dataset.csv", positive_class=2)

    report = report_for(tmp_path)

    assert not report.is_valid
    assert any("positive_class 2 was not found" in error for error in report.errors)


def test_relative_data_path_is_resolved_from_config_directory(tmp_path: Path) -> None:
    project = tmp_path / "nested" / "project"
    project.mkdir(parents=True)
    config_path = write_config(project, data_path="../shared.csv")

    config = load_config(config_path)

    assert config.data.path == (tmp_path / "nested" / "shared.csv").resolve()


def test_malformed_yaml_is_a_config_error(tmp_path: Path) -> None:
    config_path = tmp_path / "mltool.yaml"
    config_path.write_text("project: [broken", encoding="utf-8")

    with pytest.raises(ConfigError, match="malformed YAML"):
        load_config(config_path)


def test_positive_class_rejected_for_non_binary_task(tmp_path: Path) -> None:
    config = {
        "schema_version": "0.1",
        "project": {"name": "test"},
        "task": {"type": "multiclass", "target": "label", "positive_class": 1},
        "data": {"format": "csv", "path": "data.csv"},
    }
    config_path = tmp_path / "mltool.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigError, match="only valid for binary"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        (None, "schema_version", 0.1, "must be the string"),
        ("project", "name", "", "project.name"),
        ("task", "type", "clustering", "unsupported task type"),
        ("data", "format", "json", "unsupported data format"),
    ],
)
def test_invalid_configuration_values_are_rejected(
    tmp_path: Path,
    section: str | None,
    key: str,
    value: object,
    message: str,
) -> None:
    config = {
        "schema_version": "0.1",
        "project": {"name": "test"},
        "task": {"type": "binary", "target": "label"},
        "data": {"format": "csv", "path": "data.csv"},
    }
    target = config if section is None else config[section]
    target[key] = value
    config_path = tmp_path / "mltool.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(config_path)


def test_auto_format_rejects_unknown_extension(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "dataset.json").write_text("[]", encoding="utf-8")
    config_path = write_config(tmp_path, data_path="dataset.json")

    assert validate_project(config_path) == 2
    output = capsys.readouterr().out
    assert "cannot infer data format" in output
    assert "Result: INVALID" in output


def test_validate_project_returns_zero_and_prints_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pd.DataFrame({"feature": [1, 2], "label": [0, 1]}).to_csv(
        tmp_path / "dataset.csv", index=False
    )
    config_path = write_config(tmp_path, data_path="dataset.csv")

    assert validate_project(config_path) == 0
    output = capsys.readouterr().out
    assert "SHA256:" in output
    assert "Result: VALID" in output
