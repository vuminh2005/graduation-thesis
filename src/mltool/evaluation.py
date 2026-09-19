"""Task-aware validation metrics computed by MLTool."""

from __future__ import annotations

import math
from numbers import Number
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from mltool.config import EvaluationConfig


class EvaluationError(ValueError):
    """An expected prediction/metric compatibility problem."""


def _labels_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, Number) and isinstance(right, Number):
        return bool(left == right)
    return type(left) is type(right) and bool(left == right)


def _probability_column(probabilities: pd.DataFrame, label: Any) -> pd.Series:
    matches = [column for column in probabilities.columns if _labels_equal(column, label)]
    if len(matches) != 1:
        raise EvaluationError(
            f"prediction probabilities do not contain exactly one column for class {label!r}"
        )
    return probabilities[matches[0]]


def _require_probabilities(probabilities: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(probabilities, pd.DataFrame):
        raise EvaluationError("configured metrics require class probabilities")
    if probabilities.columns.duplicated().any():
        raise EvaluationError("prediction probabilities contain duplicate class columns")
    return probabilities


def _log_loss(probabilities: pd.DataFrame, y_true: pd.Series) -> float:
    """Align columns to sklearn's documented sorted-label convention."""
    try:
        labels = sorted(probabilities.columns.tolist())
    except TypeError as exc:
        raise EvaluationError(
            "class labels cannot be ordered for log_loss probability alignment"
        ) from exc
    aligned = probabilities.loc[:, labels]
    return float(log_loss(y_true, aligned, labels=labels))


def evaluate_predictions(
    *,
    task_type: str,
    evaluation: EvaluationConfig,
    y_true: pd.Series,
    predictions: pd.Series,
    probabilities: pd.DataFrame | None = None,
    positive_class: Any | None = None,
) -> dict[str, float]:
    """Compute only configured metrics against MLTool's validation split."""
    if len(y_true) != len(predictions):
        raise EvaluationError("prediction row count does not match validation target")
    metrics: dict[str, float] = {}

    if task_type == "binary":
        if positive_class is None:
            raise EvaluationError("binary evaluation could not determine the positive class")
        for metric in evaluation.metrics:
            if metric == "accuracy":
                value = accuracy_score(y_true, predictions)
            elif metric == "f1":
                value = f1_score(
                    y_true,
                    predictions,
                    pos_label=positive_class,
                    zero_division=0,
                )
            elif metric == "roc_auc":
                proba = _require_probabilities(probabilities)
                binary_truth = y_true.map(
                    lambda value: 1 if _labels_equal(value, positive_class) else 0
                )
                value = roc_auc_score(binary_truth, _probability_column(proba, positive_class))
            elif metric == "log_loss":
                proba = _require_probabilities(probabilities)
                value = _log_loss(proba, y_true)
            else:  # Config validation prevents this path.
                raise EvaluationError(f'unsupported binary metric "{metric}"')
            metrics[metric] = float(value)

    elif task_type == "multiclass":
        for metric in evaluation.metrics:
            if metric == "accuracy":
                value = accuracy_score(y_true, predictions)
            elif metric == "f1_macro":
                value = f1_score(y_true, predictions, average="macro", zero_division=0)
            elif metric == "log_loss":
                proba = _require_probabilities(probabilities)
                value = _log_loss(proba, y_true)
            else:
                raise EvaluationError(f'unsupported multiclass metric "{metric}"')
            metrics[metric] = float(value)

    elif task_type == "regression":
        for metric in evaluation.metrics:
            if metric == "rmse":
                value = math.sqrt(mean_squared_error(y_true, predictions))
            elif metric == "mae":
                value = mean_absolute_error(y_true, predictions)
            elif metric == "r2":
                value = r2_score(y_true, predictions)
            else:
                raise EvaluationError(f'unsupported regression metric "{metric}"')
            metrics[metric] = float(value)
    else:
        raise EvaluationError(f'unsupported task type "{task_type}"')

    if not all(math.isfinite(value) for value in metrics.values()):
        raise EvaluationError("evaluation produced a non-finite metric value")
    return metrics
