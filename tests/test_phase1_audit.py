from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import mltool.cli as cli
from mltool.cli import main, validate_project
from mltool.config import DataConfig, load_config
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
        "project": {"name": "audit-project"},
        "task": task,
        "data": {"format": data_format, "path": data_path},
        "validation": {"enabled": True, "fail_on_error": True},
    }
    path = root / "mltool.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def report_for(config_path: Path):
    config = load_config(config_path)
    return validate_dataset(config, load_dataset(config.data))


def test_csv_raw_header_uses_same_delimiter_as_dataframe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("age;label\n20;0\n30;1\n", encoding="utf-8")
    config_path = write_config(
        tmp_path, data_path="data.csv", data_format="csv"
    )

    assert validate_project(config_path) == 2
    output = capsys.readouterr().out
    assert 'target column "label" was not found' in output
    assert "Result: INVALID" in output


def test_duplicate_csv_headers_after_leading_blank_lines_are_detected(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("\n\nage,age,label\n20,30,0\n30,40,1\n", encoding="utf-8")
    config_path = write_config(tmp_path, data_path="data.csv")

    report = report_for(config_path)

    assert not report.is_valid
    assert 'duplicate column names found: "age"' in report.errors


def test_duplicate_csv_headers_retain_positional_dtypes(tmp_path: Path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("age,age,label\n20,unknown,0\n30,known,1\n", encoding="utf-8")
    config_path = write_config(tmp_path, data_path="data.csv")

    report = report_for(config_path)

    assert report.dataset is not None
    assert report.dataset.schema == [
        ("age", "int64"),
        ("age", "object"),
        ("label", "int64"),
    ]


def test_duplicate_target_is_not_reported_as_a_feature(tmp_path: Path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("feature,label,label\n1,0,fixed\n2,1,fixed\n", encoding="utf-8")
    config_path = write_config(tmp_path, data_path="data.csv")

    report = report_for(config_path)

    assert not report.is_valid
    assert 'duplicate column names found: "label"' in report.errors
    assert not any("label" in warning for warning in report.warnings)


def test_duplicate_parquet_field_names_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "data.parquet"
    table = pa.Table.from_arrays(
        [pa.array([20, 30]), pa.array([1000, 2000]), pa.array([0, 1])],
        names=["age", "age", "label"],
    )
    pq.write_table(table, dataset)
    config_path = write_config(
        tmp_path, data_path="data.parquet", data_format="parquet"
    )

    assert validate_project(config_path) == 2
    output = capsys.readouterr().out
    assert 'duplicate column names found: "age"' in output
    assert "Result: INVALID" in output


def test_explicit_config_path_resolves_data_relative_to_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    data = project / "data"
    elsewhere = tmp_path / "elsewhere"
    data.mkdir(parents=True)
    elsewhere.mkdir()
    pd.DataFrame({"feature": [1, 2], "label": [0, 1]}).to_csv(
        data / "dataset.csv", index=False
    )
    config_path = write_config(project)
    monkeypatch.chdir(elsewhere)

    assert validate_project(config_path) == 0
    assert "Result: VALID" in capsys.readouterr().out

    # The CLI intentionally does not discover configs in parent/other folders.
    assert main(["validate"]) == 2
    output = capsys.readouterr().out
    assert f"configuration file not found: {elsewhere / 'mltool.yaml'}" in output


@pytest.mark.parametrize(
    ("positive_class", "labels", "expected_valid"),
    [
        (1, [0, 1], True),
        ("1", ["0", "1"], True),
        ("1", [0, 1], False),
        (False, [0, 1], False),
    ],
)
def test_positive_class_matching_is_type_aware(
    tmp_path: Path,
    positive_class: object,
    labels: list[object],
    expected_valid: bool,
) -> None:
    dataset = tmp_path / "data.parquet"
    pd.DataFrame({"feature": [10, 20], "label": labels}).to_parquet(
        dataset, index=False
    )
    config_path = write_config(
        tmp_path,
        data_path="data.parquet",
        positive_class=positive_class,
    )

    report = report_for(config_path)

    assert report.is_valid is expected_valid
    if not expected_valid:
        assert any("positive_class" in error for error in report.errors)


def test_regression_accepts_fully_numeric_looking_strings(tmp_path: Path) -> None:
    dataset = tmp_path / "data.parquet"
    pd.DataFrame({"feature": [1, 2], "price": ["1.25", "-3"]}).to_parquet(
        dataset, index=False
    )
    config_path = write_config(
        tmp_path,
        task_type="regression",
        target="price",
        data_path="data.parquet",
    )

    assert report_for(config_path).is_valid


@pytest.mark.parametrize(
    ("contents", "expected_error"),
    [
        (b"", "dataset has zero rows"),
        (b"feature,label\n", "dataset has zero rows"),
        (b'feature,label\n1,"unterminated\n', "could not read CSV dataset"),
    ],
)
def test_empty_and_malformed_csv_are_expected_data_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    contents: bytes,
    expected_error: str,
) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_bytes(contents)
    config_path = write_config(
        tmp_path, data_path="data.csv", data_format="csv"
    )

    assert validate_project(config_path) == 2
    output = capsys.readouterr().out
    assert expected_error in output
    assert "Result: INVALID" in output
    assert "Traceback" not in output


def test_header_only_dataset_has_no_constant_feature_warning(tmp_path: Path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text("feature,label\n", encoding="utf-8")
    config_path = write_config(tmp_path, data_path="data.csv")

    report = report_for(config_path)

    assert "dataset has zero rows" in report.errors
    assert not any("is constant" in warning for warning in report.warnings)


def test_corrupt_parquet_is_expected_data_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "data.parquet"
    dataset.write_bytes(b"not a parquet file")
    config_path = write_config(
        tmp_path, data_path="data.parquet", data_format="parquet"
    )

    assert validate_project(config_path) == 2
    output = capsys.readouterr().out
    assert "could not read Parquet dataset" in output
    assert "Result: INVALID" in output
    assert "Traceback" not in output


def test_zero_row_parquet_is_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "data.parquet"
    pd.DataFrame(
        {
            "feature": pd.Series(dtype="int64"),
            "label": pd.Series(dtype="int64"),
        }
    ).to_parquet(dataset, index=False)
    config_path = write_config(tmp_path, data_path="data.parquet")

    assert validate_project(config_path) == 2
    output = capsys.readouterr().out
    assert "dataset has zero rows" in output
    assert "Result: INVALID" in output


def test_explicit_format_selects_parser_even_when_extension_differs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "looks-like.csv"
    dataset.write_text("feature,label\n1,0\n2,1\n", encoding="utf-8")
    config_path = write_config(
        tmp_path, data_path="looks-like.csv", data_format="parquet"
    )

    assert validate_project(config_path) == 2
    output = capsys.readouterr().out
    assert "could not read Parquet dataset" in output
    assert "Result: INVALID" in output


def test_fingerprint_uses_bytes_changes_with_file_and_does_not_modify_source(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    original = b"feature,label\n1,0\n2,1\n"
    first.write_bytes(original)
    second.write_bytes(original)

    first_loaded = load_dataset(DataConfig(path=first, format="csv"))
    second_loaded = load_dataset(DataConfig(path=second, format="csv"))

    expected = hashlib.sha256(original).hexdigest()
    assert first_loaded.fingerprint == expected
    assert second_loaded.fingerprint == expected
    assert first.read_bytes() == original

    modified = original + b"3,0\n"
    first.write_bytes(modified)
    changed = load_dataset(DataConfig(path=first, format="csv"))
    assert changed.fingerprint == hashlib.sha256(modified).hexdigest()
    assert changed.fingerprint != expected
    assert first.read_bytes() == modified


def test_class_imbalance_warning_is_nonfatal_and_target_is_not_a_feature(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "data.csv"
    pd.DataFrame({"feature": range(20), "label": [0] * 19 + [1]}).to_csv(
        dataset, index=False
    )
    config_path = write_config(tmp_path, data_path="data.csv")

    report = report_for(config_path)

    assert report.is_valid
    assert "class 1 represents only 5.0% of non-null target values" in report.warnings
    assert not any('feature "label"' in warning for warning in report.warnings)


def test_multiclass_imbalance_warning_is_deterministic_and_nonfatal(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "data.csv"
    labels = ["major"] * 18 + ["rare-a", "rare-b"]
    pd.DataFrame({"feature": range(20), "label": labels}).to_csv(
        dataset, index=False
    )
    config_path = write_config(
        tmp_path,
        task_type="multiclass",
        data_path="data.csv",
    )

    report = report_for(config_path)

    assert report.is_valid
    assert report.warnings == [
        "class 'rare-a' represents only 5.0% of non-null target values"
    ]


def test_unexpected_internal_error_returns_one_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_unexpectedly(path: Path) -> None:
        raise RuntimeError("internal probe")

    monkeypatch.setattr(cli, "load_config", fail_unexpectedly)

    assert main(["validate"]) == 1
    captured = capsys.readouterr()
    assert "MLTool failed unexpectedly: internal probe" in captured.err
    assert "Traceback" not in captured.err


def test_python_module_entrypoint_without_command_exits_two(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "mltool"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "usage: mltool" in completed.stderr
    assert "Traceback" not in completed.stderr
