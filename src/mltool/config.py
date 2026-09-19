"""Load and validate the intentionally small Phase-1/2 configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_TASKS = {"binary", "multiclass", "regression"}
SUPPORTED_FORMATS = {"auto", "csv", "parquet"}


class ConfigError(ValueError):
    """An understandable problem with an MLTool configuration."""


@dataclass(frozen=True)
class ProjectConfig:
    name: str


@dataclass(frozen=True)
class TaskConfig:
    type: str
    target: str
    positive_class: Any | None = None


@dataclass(frozen=True)
class DataConfig:
    path: Path
    format: str


@dataclass(frozen=True)
class ValidationConfig:
    enabled: bool = True
    fail_on_error: bool = True


@dataclass(frozen=True)
class SplitConfig:
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify: str | bool = "auto"
    random_seed: int = 42


@dataclass(frozen=True)
class ExternalPreprocessingConfig:
    enabled: bool = False
    entrypoint: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessingConfig:
    external: ExternalPreprocessingConfig


@dataclass(frozen=True)
class MLToolConfig:
    schema_version: str
    project: ProjectConfig
    task: TaskConfig
    data: DataConfig
    validation: ValidationConfig
    split: SplitConfig
    preprocessing: PreprocessingConfig
    config_path: Path


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f'configuration section "{key}" is required and must be a mapping')
    return value


def _non_empty_string(parent: dict[str, Any], key: str, qualified_name: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f'"{qualified_name}" is required and must be a non-empty string')
    return value.strip()


def _boolean(parent: dict[str, Any], key: str, default: bool, qualified_name: str) -> bool:
    value = parent.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f'"{qualified_name}" must be true or false')
    return value


def _ratio(parent: dict[str, Any], key: str, default: float) -> float:
    value = parent.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f'"split.{key}" must be a number between 0 and 1')
    result = float(value)
    if not 0 < result < 1:
        raise ConfigError(f'"split.{key}" must be greater than 0 and less than 1')
    return result


def load_config(path: Path | str = Path("mltool.yaml")) -> MLToolConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f'configuration file not found: {config_path}')

    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        detail = str(exc).splitlines()[0]
        raise ConfigError(f"malformed YAML in {config_path}: {detail}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read configuration {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")

    schema_version = raw.get("schema_version")
    if schema_version != "0.1":
        raise ConfigError('"schema_version" must be the string "0.1"')

    project_raw = _mapping(raw, "project")
    task_raw = _mapping(raw, "task")
    data_raw = _mapping(raw, "data")

    project_name = _non_empty_string(project_raw, "name", "project.name")
    task_type = _non_empty_string(task_raw, "type", "task.type")
    if task_type not in SUPPORTED_TASKS:
        supported = ", ".join(sorted(SUPPORTED_TASKS))
        raise ConfigError(f'unsupported task type "{task_type}"; expected one of: {supported}')
    target = _non_empty_string(task_raw, "target", "task.target")

    has_positive_class = "positive_class" in task_raw
    positive_class = task_raw.get("positive_class")
    if has_positive_class and task_type != "binary":
        raise ConfigError('"task.positive_class" is only valid for binary tasks')
    if has_positive_class and positive_class is None:
        raise ConfigError('"task.positive_class" may not be null when configured')

    data_path_text = _non_empty_string(data_raw, "path", "data.path")
    data_format = _non_empty_string(data_raw, "format", "data.format")
    if data_format not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ConfigError(f'unsupported data format "{data_format}"; expected one of: {supported}')
    data_path = (config_path.parent / data_path_text).resolve()

    validation_raw = raw.get("validation", {})
    if not isinstance(validation_raw, dict):
        raise ConfigError('configuration section "validation" must be a mapping')
    validation = ValidationConfig(
        enabled=_boolean(validation_raw, "enabled", True, "validation.enabled"),
        fail_on_error=_boolean(
            validation_raw, "fail_on_error", True, "validation.fail_on_error"
        ),
    )

    split_raw = raw.get("split", {})
    if not isinstance(split_raw, dict):
        raise ConfigError('configuration section "split" must be a mapping')
    validation_ratio = _ratio(split_raw, "validation_ratio", 0.15)
    test_ratio = _ratio(split_raw, "test_ratio", 0.15)
    if validation_ratio + test_ratio >= 1:
        raise ConfigError(
            '"split.validation_ratio" plus "split.test_ratio" must be less than 1'
        )
    stratify = split_raw.get("stratify", "auto")
    if not isinstance(stratify, bool) and stratify != "auto":
        raise ConfigError('"split.stratify" must be one of: auto, true, false')
    random_seed = split_raw.get("random_seed", 42)
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ConfigError('"split.random_seed" must be an integer')
    split = SplitConfig(
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        stratify=stratify,
        random_seed=random_seed,
    )

    preprocessing_raw = raw.get("preprocessing", {})
    if not isinstance(preprocessing_raw, dict):
        raise ConfigError('configuration section "preprocessing" must be a mapping')
    external_raw = preprocessing_raw.get("external", {})
    if not isinstance(external_raw, dict):
        raise ConfigError('"preprocessing.external" must be a mapping')
    external_enabled = _boolean(
        external_raw, "enabled", False, "preprocessing.external.enabled"
    )
    entrypoint = external_raw.get("entrypoint")
    if entrypoint is not None and (
        not isinstance(entrypoint, str) or not entrypoint.strip()
    ):
        raise ConfigError(
            '"preprocessing.external.entrypoint" must be null or a non-empty string'
        )
    if external_enabled and entrypoint is None:
        raise ConfigError(
            '"preprocessing.external.entrypoint" is required when external preprocessing is enabled'
        )
    params = external_raw.get("params", {})
    if not isinstance(params, dict) or not all(isinstance(key, str) for key in params):
        raise ConfigError('"preprocessing.external.params" must be a mapping with string keys')
    try:
        json.dumps(params)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            '"preprocessing.external.params" values must be JSON-serializable'
        ) from exc
    preprocessing = PreprocessingConfig(
        external=ExternalPreprocessingConfig(
            enabled=external_enabled,
            entrypoint=entrypoint.strip() if isinstance(entrypoint, str) else None,
            params=dict(params),
        )
    )

    return MLToolConfig(
        schema_version=schema_version,
        project=ProjectConfig(name=project_name),
        task=TaskConfig(
            type=task_type,
            target=target,
            positive_class=positive_class if has_positive_class else None,
        ),
        data=DataConfig(path=data_path, format=data_format),
        validation=validation,
        split=split,
        preprocessing=preprocessing,
        config_path=config_path,
    )
