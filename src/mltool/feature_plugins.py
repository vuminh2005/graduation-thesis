"""Loading and executing independent Phase-3 feature plugins."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from mltool.config import FeaturePluginConfig, execute_source, source_sha256


class FeaturePluginError(ValueError):
    """An expected feature-plugin loading, execution, or contract failure."""


@dataclass
class GeneratedFeatureFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    generated_columns: list[str]
    resolved_entrypoint: str


def _resolve_entrypoint(entrypoint: str, config_path: Path) -> tuple[Path, str]:
    file_text, separator, class_name = entrypoint.rpartition(":")
    if not separator or not file_text.strip() or not class_name.strip():
        raise FeaturePluginError(
            'feature plugin entrypoint must use "<python-file>:<class-name>"'
        )
    plugin_path = Path(file_text.strip()).expanduser()
    if not plugin_path.is_absolute():
        plugin_path = config_path.parent / plugin_path
    plugin_path = plugin_path.resolve()
    if not plugin_path.is_file():
        raise FeaturePluginError(f"feature plugin file not found: {plugin_path}")
    return plugin_path, class_name.strip()


def plugin_source_sha256(spec: FeaturePluginConfig, config_path: Path) -> str | None:
    """SHA-256 of the plugin's entrypoint file, or None when it cannot be read.

    Only that one file is hashed: a sibling module it imports is not tracked.
    """
    try:
        path, _ = _resolve_entrypoint(spec.entrypoint, config_path)
    except FeaturePluginError:
        return None
    return source_sha256(path)


def load_feature_plugin(
    spec: FeaturePluginConfig,
    config_path: Path,
) -> tuple[Any, str]:
    plugin_path, class_name = _resolve_entrypoint(spec.entrypoint, config_path)
    identity = f"{plugin_path}:{class_name}:{spec.name}"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    module_name = f"_mltool_feature_plugin_{suffix}"
    module_spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    if module_spec is None or module_spec.loader is None:
        raise FeaturePluginError(f"could not create a module loader for {plugin_path}")
    module = importlib.util.module_from_spec(module_spec)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        execute_source(module, plugin_path)
    except (Exception, SystemExit) as exc:
        raise FeaturePluginError(
            f"could not load feature plugin module {plugin_path}: {exc}"
        ) from exc
    finally:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module

    plugin_class = module.__dict__.get(class_name)
    if plugin_class is None:
        raise FeaturePluginError(
            f'feature plugin class "{class_name}" was not found in {plugin_path}'
        )
    if not inspect.isclass(plugin_class):
        raise FeaturePluginError(f'feature plugin "{class_name}" is not a class')
    try:
        instance = plugin_class(**spec.params)
    except (Exception, SystemExit) as exc:
        raise FeaturePluginError(
            f'could not instantiate feature plugin "{spec.name}": {exc}'
        ) from exc
    if not callable(getattr(instance, "fit", None)):
        raise FeaturePluginError(
            f'feature plugin "{spec.name}" must provide callable fit(X_train)'
        )
    if not callable(getattr(instance, "transform", None)):
        raise FeaturePluginError(
            f'feature plugin "{spec.name}" must provide callable transform(X)'
        )
    return instance, f"{plugin_path}:{class_name}"


def _transform(
    plugin: Any,
    features: pd.DataFrame,
    *,
    plugin_name: str,
    split_name: str,
    target: str,
) -> pd.DataFrame:
    try:
        generated = plugin.transform(features.copy(deep=True))
    except (Exception, SystemExit) as exc:
        raise FeaturePluginError(
            f'feature plugin "{plugin_name}" transform failed for split "{split_name}": {exc}'
        ) from exc
    if not isinstance(generated, pd.DataFrame):
        raise FeaturePluginError(
            f'feature plugin "{plugin_name}" transform for split "{split_name}" '
            "must return a pandas DataFrame"
        )
    if len(generated) != len(features):
        raise FeaturePluginError(
            f'feature plugin "{plugin_name}" changed row count for split "{split_name}": '
            f"expected {len(features)}, got {len(generated)}"
        )
    if not generated.index.equals(features.index):
        raise FeaturePluginError(
            f'feature plugin "{plugin_name}" changed or reordered the index '
            f'for split "{split_name}"'
        )
    if generated.shape[1] == 0:
        raise FeaturePluginError(
            f'feature plugin "{plugin_name}" generated zero feature columns '
            f'for split "{split_name}"'
        )
    if not all(isinstance(column, str) and column for column in generated.columns):
        raise FeaturePluginError(
            f'feature plugin "{plugin_name}" generated non-string or empty column names '
            f'for split "{split_name}"'
        )
    duplicates = generated.columns[generated.columns.duplicated()].tolist()
    if duplicates:
        names = ", ".join(repr(name) for name in dict.fromkeys(duplicates))
        raise FeaturePluginError(
            f'feature plugin "{plugin_name}" generated duplicate columns '
            f'for split "{split_name}": {names}'
        )
    if target in generated.columns:
        raise FeaturePluginError(
            f'feature plugin "{plugin_name}" generated configured target column '
            f'"{target}" for split "{split_name}"'
        )
    return generated.copy(deep=True)


def fit_and_generate_features(
    spec: FeaturePluginConfig,
    *,
    config_path: Path,
    target: str,
    fit_split: str,
    base_frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[str], str, Any]:
    """Instantiate, fit once on ``fit_split``, and transform every base frame.

    Shared by Phase 3 (fit on train) and Phase 6 (refit on train+validation).
    The fitted plugin instance is returned so Phase 6 can persist its state.
    """
    plugin, resolved_entrypoint = load_feature_plugin(spec, config_path)
    try:
        plugin.fit(base_frames[fit_split].copy(deep=True))
    except (Exception, SystemExit) as exc:
        raise FeaturePluginError(
            f'feature plugin "{spec.name}" fit failed on {fit_split} features: {exc}'
        ) from exc

    frames = {
        split_name: _transform(
            plugin,
            base,
            plugin_name=spec.name,
            split_name=split_name,
            target=target,
        )
        for split_name, base in base_frames.items()
    }
    expected_columns = frames[fit_split].columns.tolist()
    for split_name, generated in frames.items():
        actual_columns = generated.columns.tolist()
        if actual_columns != expected_columns:
            raise FeaturePluginError(
                f'feature plugin "{spec.name}" generated inconsistent columns: '
                f"{fit_split} has {expected_columns!r}, {split_name} has {actual_columns!r}"
            )
    return frames, expected_columns, resolved_entrypoint, plugin


def generate_features(
    spec: FeaturePluginConfig,
    *,
    config_path: Path,
    target: str,
    train_base: pd.DataFrame,
    validation_base: pd.DataFrame,
    test_base: pd.DataFrame,
) -> GeneratedFeatureFrames:
    frames, columns, resolved_entrypoint, _ = fit_and_generate_features(
        spec,
        config_path=config_path,
        target=target,
        fit_split="train",
        base_frames={"train": train_base, "validation": validation_base, "test": test_base},
    )
    return GeneratedFeatureFrames(
        train=frames["train"],
        validation=frames["validation"],
        test=frames["test"],
        generated_columns=columns,
        resolved_entrypoint=resolved_entrypoint,
    )
