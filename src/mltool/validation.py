"""Simple, transparent Phase-1 validation rules."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from mltool.config import MLToolConfig
from mltool.data import LoadedDataset
from mltool.report import DatasetSummary, ValidationReport


CLASS_IMBALANCE_THRESHOLD = 0.10


def _duplicate_names(names: list[str]) -> list[str]:
    counts = Counter(names)
    return [name for name, count in counts.items() if count > 1]


def _distribution(series: pd.Series) -> list[tuple[Any, int]]:
    counts = series.value_counts(dropna=True, sort=False)
    values = [(value, int(count)) for value, count in counts.items()]
    return sorted(values, key=lambda item: str(item[0]))


def validate_dataset(config: MLToolConfig, dataset: LoadedDataset) -> ValidationReport:
    frame = dataset.frame
    report = ValidationReport(
        project_name=config.project.name,
        task_type=config.task.type,
        target=config.task.target,
        dataset=DatasetSummary(
            path=dataset.path,
            format=dataset.format,
            rows=dataset.row_count,
            columns=dataset.column_count,
            fingerprint=dataset.fingerprint,
            schema=[
                (
                    str(name),
                    dataset.dtypes[index] if index < len(dataset.dtypes) else "unknown",
                )
                for index, name in enumerate(dataset.column_names)
            ],
        ),
    )

    if dataset.row_count == 0:
        report.errors.append("dataset has zero rows")
    if dataset.column_count == 0:
        report.errors.append("dataset has zero columns")

    duplicates = _duplicate_names(dataset.column_names)
    if duplicates:
        names = ", ".join(f'"{name}"' for name in duplicates)
        report.errors.append(f"duplicate column names found: {names}")

    for column in config.data.id_columns:
        if column not in dataset.column_names:
            report.errors.append(f'id column "{column}" (data.id_columns) was not found')

    target_name = config.task.target
    if target_name not in dataset.column_names:
        report.errors.append(f'target column "{target_name}" was not found')
        _add_feature_warnings(report, frame, target_name=None)
        return report

    # A duplicate target cannot be selected unambiguously; the duplicate-name
    # error above is sufficient. Feature diagnostics would mislabel the
    # mangled copies of the target as features, so skip them as well.
    if dataset.column_names.count(target_name) > 1:
        return report

    target = frame[target_name]
    null_count = int(target.isna().sum())
    if len(target) > 0 and null_count == len(target):
        report.errors.append(f'target column "{target_name}" is entirely null')
    elif null_count:
        report.errors.append(
            f'target column "{target_name}" contains {null_count} null value(s)'
        )

    non_null_target = target.dropna()
    unique_count = int(non_null_target.nunique(dropna=True))

    if config.task.type == "binary":
        if unique_count != 2:
            report.errors.append(
                f'binary target "{target_name}" must contain exactly 2 distinct non-null classes; '
                f"found {unique_count}"
            )
        if config.task.positive_class is not None:
            class_values = non_null_target.unique().tolist()
            if not any(
                _class_values_equal(config.task.positive_class, value)
                for value in class_values
            ):
                report.errors.append(
                    f'configured positive_class {config.task.positive_class!r} was not found '
                    f'in target "{target_name}"'
                )
        report.target_distribution = _distribution(non_null_target)
        report.target_kind = str(unique_count)
        _add_imbalance_warning(report, report.target_distribution, len(non_null_target))
    elif config.task.type == "multiclass":
        if unique_count < 2:
            report.errors.append(
                f'multiclass target "{target_name}" must contain at least 2 distinct non-null classes; '
                f"found {unique_count}"
            )
        report.target_distribution = _distribution(non_null_target)
        report.target_kind = str(unique_count)
        _add_imbalance_warning(report, report.target_distribution, len(non_null_target))
    else:
        if not _is_usable_numeric(non_null_target):
            report.errors.append(
                f'regression target "{target_name}" is not usable as numeric data'
            )

    _add_feature_warnings(report, frame, target_name=target_name)
    return report


def _is_usable_numeric(series: pd.Series) -> bool:
    if series.empty or is_bool_dtype(series.dtype):
        return False
    if is_numeric_dtype(series.dtype):
        return True
    converted = pd.to_numeric(series, errors="coerce")
    return bool(converted.notna().all())


def _class_values_equal(configured: Any, observed: Any) -> bool:
    """Compare class labels without equating booleans with numeric 0 or 1."""
    configured_is_bool = isinstance(configured, bool)
    observed_is_bool = isinstance(observed, bool)
    if configured_is_bool != observed_is_bool:
        return False
    return bool(configured == observed)


def _add_imbalance_warning(
    report: ValidationReport,
    distribution: list[tuple[Any, int]],
    target_count: int,
) -> None:
    if len(distribution) < 2 or target_count == 0:
        return
    minority_value, minority_count = min(distribution, key=lambda item: item[1])
    share = minority_count / target_count
    if share < CLASS_IMBALANCE_THRESHOLD:
        report.warnings.append(
            f'class {minority_value!r} represents only {share:.1%} of non-null target values'
        )


def _add_feature_warnings(
    report: ValidationReport,
    frame: pd.DataFrame,
    target_name: str | None,
) -> None:
    if target_name is not None and not any(name != target_name for name in frame.columns):
        report.warnings.append(
            "dataset has no columns other than the target; there are no features to "
            "build a FeatureSet from"
        )

    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        report.warnings.append(f"{duplicate_rows} duplicate row(s) found")

    for name in frame.columns:
        if target_name is not None and name == target_name:
            continue
        series = frame[name]
        missing = int(series.isna().sum())
        if missing:
            report.warnings.append(
                f'feature "{name}" contains {missing} missing value(s)'
            )
        if len(series) > 0 and series.nunique(dropna=False) <= 1:
            report.warnings.append(f'feature "{name}" is constant')
