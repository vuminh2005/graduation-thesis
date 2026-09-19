"""Load and validate the intentionally small Phase-1 configuration schema."""

from __future__ import annotations

from dataclasses import dataclass
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
class MLToolConfig:
    schema_version: str
    project: ProjectConfig
    task: TaskConfig
    data: DataConfig
    validation: ValidationConfig
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
        config_path=config_path,
    )

