"""Deterministic Phase-2 train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
from sklearn.model_selection import train_test_split

from mltool.config import SplitConfig


class SplitError(ValueError):
    """An expected dataset/configuration problem that prevents splitting."""


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    stratified: bool


def should_stratify(task_type: str, configured: str | bool) -> bool:
    if isinstance(configured, bool):
        return configured
    return task_type in {"binary", "multiclass"}


def split_dataset(
    frame: pd.DataFrame,
    *,
    target: str,
    task_type: str,
    config: SplitConfig,
) -> DatasetSplits:
    """Split using ratios relative to the original dataset.

    Test is separated first. Both holdout row counts are calculated from their
    requested original-dataset fractions, avoiding floating-point drift in the
    equivalent sequential ratio V / (1 - T).
    """
    stratified = should_stratify(task_type, config.stratify)
    first_stratify = frame[target] if stratified else None
    total_rows = len(frame)
    test_rows = math.ceil(config.test_ratio * total_rows)
    validation_rows = math.ceil(config.validation_ratio * total_rows)

    try:
        train_validation, test = train_test_split(
            frame,
            test_size=test_rows,
            random_state=config.random_seed,
            shuffle=True,
            stratify=first_stratify,
        )
        second_stratify = train_validation[target] if stratified else None
        train, validation = train_test_split(
            train_validation,
            test_size=validation_rows,
            random_state=config.random_seed,
            shuffle=True,
            stratify=second_stratify,
        )
    except ValueError as exc:
        if stratified:
            raise SplitError(
                "stratified split is impossible for the configured ratios and class counts: "
                f"{exc}"
            ) from exc
        raise SplitError(f"dataset cannot be split with the configured ratios: {exc}") from exc

    return DatasetSplits(
        train=train.copy(deep=True),
        validation=validation.copy(deep=True),
        test=test.copy(deep=True),
        stratified=stratified,
    )
