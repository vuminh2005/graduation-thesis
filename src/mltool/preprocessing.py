"""Small external-preprocessor loader and leakage-safe execution contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from mltool.config import ExternalPreprocessingConfig
from mltool.splitting import DatasetSplits


class PreprocessingError(ValueError):
    """An expected plugin loading, execution, or contract failure."""


@dataclass
class PreprocessedSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    fitted_preprocessor: Any | None = None
    preprocessor_name: str | None = None
    resolved_entrypoint: str | None = None


def _resolve_entrypoint(entrypoint: str, config_path: Path) -> tuple[Path, str]:
    file_text, separator, class_name = entrypoint.rpartition(":")
    if not separator or not file_text.strip() or not class_name.strip():
        raise PreprocessingError(
            'external preprocessor entrypoint must use "<python-file>:<class-name>"'
        )
    plugin_path = Path(file_text.strip()).expanduser()
    if not plugin_path.is_absolute():
        plugin_path = config_path.parent / plugin_path
    plugin_path = plugin_path.resolve()
    if not plugin_path.is_file():
        raise PreprocessingError(f"external preprocessor file not found: {plugin_path}")
    return plugin_path, class_name.strip()


def load_external_preprocessor(
    config: ExternalPreprocessingConfig,
    config_path: Path,
) -> tuple[Any, str, str]:
    if config.entrypoint is None:  # Config validation normally prevents this.
        raise PreprocessingError("external preprocessor entrypoint is required")
    plugin_path, class_name = _resolve_entrypoint(config.entrypoint, config_path)
    module_suffix = hashlib.sha256(str(plugin_path).encode("utf-8")).hexdigest()[:16]
    module_name = f"_mltool_external_preprocessor_{module_suffix}"
    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    if spec is None or spec.loader is None:
        raise PreprocessingError(f"could not create a module loader for {plugin_path}")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except (Exception, SystemExit) as exc:
        raise PreprocessingError(
            f"could not load external preprocessor module {plugin_path}: {exc}"
        ) from exc
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module

    preprocessor_class = module.__dict__.get(class_name)
    if preprocessor_class is None:
        raise PreprocessingError(
            f'external preprocessor class "{class_name}" was not found in {plugin_path}'
        )
    if not inspect.isclass(preprocessor_class):
        raise PreprocessingError(
            f'external preprocessor "{class_name}" is not a class'
        )
    try:
        instance = preprocessor_class(**config.params)
    except (Exception, SystemExit) as exc:
        raise PreprocessingError(
            f'could not instantiate external preprocessor "{class_name}": {exc}'
        ) from exc
    if not callable(getattr(instance, "fit", None)):
        raise PreprocessingError(
            f'external preprocessor "{class_name}" must provide callable fit(X_train)'
        )
    if not callable(getattr(instance, "transform", None)):
        raise PreprocessingError(
            f'external preprocessor "{class_name}" must provide callable transform(X)'
        )
    resolved_entrypoint = f"{plugin_path}:{class_name}"
    return instance, class_name, resolved_entrypoint


def _transform_one(
    preprocessor: Any,
    features: pd.DataFrame,
    *,
    split_name: str,
    target: str,
) -> pd.DataFrame:
    transform_input = features.copy(deep=True)
    try:
        transformed = preprocessor.transform(transform_input)
    except (Exception, SystemExit) as exc:
        raise PreprocessingError(
            f'external preprocessor transform failed for split "{split_name}": {exc}'
        ) from exc
    if not isinstance(transformed, pd.DataFrame):
        raise PreprocessingError(
            f'external preprocessor transform for split "{split_name}" must return a pandas DataFrame'
        )
    if len(transformed) != len(features):
        raise PreprocessingError(
            f'external preprocessor changed row count for split "{split_name}": '
            f"expected {len(features)}, got {len(transformed)}"
        )
    if not transformed.index.equals(features.index):
        raise PreprocessingError(
            f'external preprocessor changed or reordered the index for split "{split_name}"'
        )
    duplicates = transformed.columns[transformed.columns.duplicated()].tolist()
    if duplicates:
        names = ", ".join(repr(name) for name in dict.fromkeys(duplicates))
        raise PreprocessingError(
            f'external preprocessor produced duplicate columns for split "{split_name}": {names}'
        )
    if target in transformed.columns:
        raise PreprocessingError(
            f'external preprocessor produced configured target column "{target}" '
            f'for split "{split_name}"'
        )
    return transformed.copy(deep=True)


def preprocess_splits(
    splits: DatasetSplits,
    *,
    target: str,
    config: ExternalPreprocessingConfig,
    config_path: Path,
) -> PreprocessedSplits:
    if not config.enabled:
        return PreprocessedSplits(
            train=splits.train.copy(deep=True),
            validation=splits.validation.copy(deep=True),
            test=splits.test.copy(deep=True),
        )

    preprocessor, class_name, resolved_entrypoint = load_external_preprocessor(
        config, config_path
    )
    features = {
        "train": splits.train.drop(columns=[target]),
        "validation": splits.validation.drop(columns=[target]),
        "test": splits.test.drop(columns=[target]),
    }
    targets = {
        "train": splits.train[target].copy(deep=True),
        "validation": splits.validation[target].copy(deep=True),
        "test": splits.test[target].copy(deep=True),
    }

    # The only fit call in Phase 2 receives a defensive copy of train features.
    try:
        preprocessor.fit(features["train"].copy(deep=True))
    except (Exception, SystemExit) as exc:
        raise PreprocessingError(
            f'external preprocessor "{class_name}" fit failed on train features: {exc}'
        ) from exc

    prepared: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validation", "test"):
        transformed = _transform_one(
            preprocessor,
            features[split_name],
            split_name=split_name,
            target=target,
        )
        transformed[target] = targets[split_name]
        prepared[split_name] = transformed

    return PreprocessedSplits(
        train=prepared["train"],
        validation=prepared["validation"],
        test=prepared["test"],
        fitted_preprocessor=preprocessor,
        preprocessor_name=class_name,
        resolved_entrypoint=resolved_entrypoint,
    )
