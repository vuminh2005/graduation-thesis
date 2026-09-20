"""Load and validate the intentionally small Phase-1 through Phase-5 schema."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import yaml


SUPPORTED_TASKS = {"binary", "multiclass", "regression"}
SUPPORTED_FORMATS = {"auto", "csv", "parquet"}
SUPPORTED_MODEL_FAMILIES = {"GBM", "CAT", "XGB", "RF", "XT"}
SUPPORTED_METRICS = {
    "binary": {"roc_auc", "f1", "accuracy", "log_loss"},
    "multiclass": {"accuracy", "f1_macro", "log_loss"},
    "regression": {"rmse", "mae", "r2"},
}
DEFAULT_METRICS = {
    "binary": ("roc_auc", ["f1", "accuracy"]),
    "multiclass": ("accuracy", ["f1_macro"]),
    "regression": ("rmse", ["mae", "r2"]),
}
METRIC_DIRECTIONS = {
    "roc_auc": "maximize",
    "f1": "maximize",
    "accuracy": "maximize",
    "f1_macro": "maximize",
    "r2": "maximize",
    "log_loss": "minimize",
    "rmse": "minimize",
    "mae": "minimize",
}
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
SAFE_FEATURE_SET_NAME = SAFE_IDENTIFIER
SAFE_PLUGIN_NAME = SAFE_IDENTIFIER


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
class FeaturePluginConfig:
    name: str
    entrypoint: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureSetConfig:
    name: str
    source_columns: list[str]
    plugins: list[str]


@dataclass(frozen=True)
class FeaturesConfig:
    plugins: list[FeaturePluginConfig]
    sets: list[FeatureSetConfig]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationConfig:
    primary_metric: str
    secondary_metrics: list[str]

    @property
    def metrics(self) -> list[str]:
        return [self.primary_metric, *self.secondary_metrics]

    @property
    def direction(self) -> str:
        return METRIC_DIRECTIONS[self.primary_metric]


@dataclass(frozen=True)
class TrainingConfig:
    time_limit_seconds: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class HpoConfig:
    top_n: int
    num_trials: int
    time_limit_seconds: int


@dataclass(frozen=True)
class MLToolConfig:
    schema_version: str
    project: ProjectConfig
    task: TaskConfig
    data: DataConfig
    validation: ValidationConfig
    split: SplitConfig
    preprocessing: PreprocessingConfig
    features: FeaturesConfig
    models: list[ModelConfig]
    evaluation: EvaluationConfig
    training: TrainingConfig
    config_path: Path
    hpo: HpoConfig | None = None


def _reject_unknown(section: dict[str, Any], allowed: set[str], name: str) -> None:
    """Fail on misspelled keys instead of silently dropping the intended setting."""
    unknown = sorted(str(key) for key in set(section) - allowed)
    if unknown:
        raise ConfigError(f'unsupported "{name}" setting(s): {", ".join(unknown)}')


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


def _json_mapping(value: Any, qualified_name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ConfigError(f'"{qualified_name}" must be a mapping with string keys')
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f'"{qualified_name}" values must be JSON-serializable') from exc
    return dict(value)


def _model_config(raw: dict[str, Any]) -> list[ModelConfig]:
    models_raw = raw.get("models", [])
    if not isinstance(models_raw, list):
        raise ConfigError('"models" must be a list')
    models: list[ModelConfig] = []
    names: set[str] = set()
    for index, model_raw in enumerate(models_raw):
        prefix = f"models[{index}]"
        if not isinstance(model_raw, dict):
            raise ConfigError(f'"{prefix}" must be a mapping')
        _reject_unknown(model_raw, {"name", "family", "params"}, prefix)
        name = _non_empty_string(model_raw, "name", f"{prefix}.name")
        if not SAFE_IDENTIFIER.fullmatch(name):
            raise ConfigError(
                f'model name "{name}" is unsafe; use only letters, numbers, underscore, and hyphen'
            )
        if name in names:
            raise ConfigError(f'duplicate model name "{name}"')
        names.add(name)
        family = _non_empty_string(model_raw, "family", f"{prefix}.family")
        if family not in SUPPORTED_MODEL_FAMILIES:
            supported = ", ".join(sorted(SUPPORTED_MODEL_FAMILIES))
            raise ConfigError(
                f'unsupported model family "{family}"; expected one of: {supported}'
            )
        params = _json_mapping(model_raw.get("params", {}), f"{prefix}.params")
        models.append(ModelConfig(name=name, family=family, params=params))
    return models


def _evaluation_config(raw: dict[str, Any], task_type: str) -> EvaluationConfig:
    default_primary, default_secondary = DEFAULT_METRICS[task_type]
    evaluation_raw = raw.get("evaluation", {})
    if not isinstance(evaluation_raw, dict):
        raise ConfigError('configuration section "evaluation" must be a mapping')
    _reject_unknown(evaluation_raw, {"primary_metric", "secondary_metrics"}, "evaluation")
    primary = evaluation_raw.get("primary_metric", default_primary)
    if not isinstance(primary, str) or not primary.strip():
        raise ConfigError('"evaluation.primary_metric" must be a non-empty string')
    primary = primary.strip()
    secondary = evaluation_raw.get("secondary_metrics", default_secondary)
    if not isinstance(secondary, list) or not all(
        isinstance(metric, str) and metric.strip() for metric in secondary
    ):
        raise ConfigError(
            '"evaluation.secondary_metrics" must be a list of non-empty strings'
        )
    secondary = [metric.strip() for metric in secondary]
    if len(secondary) != len(set(secondary)):
        raise ConfigError('"evaluation.secondary_metrics" must contain unique values')
    if primary in secondary:
        raise ConfigError(
            '"evaluation.secondary_metrics" must not repeat the primary metric'
        )
    allowed = SUPPORTED_METRICS[task_type]
    for metric in [primary, *secondary]:
        if metric not in allowed:
            supported = ", ".join(sorted(allowed))
            raise ConfigError(
                f'metric "{metric}" is not supported for task "{task_type}"; '
                f"expected one of: {supported}"
            )
    return EvaluationConfig(primary_metric=primary, secondary_metrics=secondary)


def _training_config(raw: dict[str, Any]) -> TrainingConfig:
    training_raw = raw.get("training", {})
    if not isinstance(training_raw, dict):
        raise ConfigError('configuration section "training" must be a mapping')
    _reject_unknown(training_raw, {"time_limit_seconds", "seed"}, "training")
    time_limit = training_raw.get("time_limit_seconds")
    if time_limit is not None and (
        isinstance(time_limit, bool) or not isinstance(time_limit, int) or time_limit <= 0
    ):
        raise ConfigError('"training.time_limit_seconds" must be null or a positive integer')
    seed = training_raw.get("seed")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31
    ):
        raise ConfigError('"training.seed" must be null or an integer in [0, 2**31)')
    return TrainingConfig(time_limit_seconds=time_limit, seed=seed)


def _positive_int(parent: dict[str, Any], key: str, default: int | None) -> int:
    value = parent.get(key, default)
    if value is None:
        raise ConfigError(f'"hpo.{key}" is required')
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f'"hpo.{key}" must be a positive integer')
    return value


def _hpo_config(raw: dict[str, Any]) -> HpoConfig | None:
    if "hpo" not in raw or raw["hpo"] is None:
        return None
    hpo_raw = raw["hpo"]
    if not isinstance(hpo_raw, dict):
        raise ConfigError('configuration section "hpo" must be a mapping')
    _reject_unknown(hpo_raw, {"top_n", "num_trials", "time_limit_seconds"}, "hpo")
    return HpoConfig(
        top_n=_positive_int(hpo_raw, "top_n", 3),
        num_trials=_positive_int(hpo_raw, "num_trials", 10),
        time_limit_seconds=_positive_int(hpo_raw, "time_limit_seconds", None),
    )


def _feature_config(raw: dict[str, Any]) -> FeaturesConfig:
    features_raw = raw.get("features")
    if features_raw is None:
        return FeaturesConfig(
            plugins=[],
            sets=[FeatureSetConfig(name="base", source_columns=["*"], plugins=[])],
        )
    if not isinstance(features_raw, dict):
        raise ConfigError('configuration section "features" must be a mapping')
    _reject_unknown(features_raw, {"plugins", "sets"}, "features")

    plugins_raw = features_raw.get("plugins", [])
    if not isinstance(plugins_raw, list):
        raise ConfigError('"features.plugins" must be a list')
    plugins: list[FeaturePluginConfig] = []
    plugin_names: set[str] = set()
    for index, plugin_raw in enumerate(plugins_raw):
        prefix = f"features.plugins[{index}]"
        if not isinstance(plugin_raw, dict):
            raise ConfigError(f'"{prefix}" must be a mapping')
        _reject_unknown(plugin_raw, {"name", "entrypoint", "params"}, prefix)
        name = _non_empty_string(plugin_raw, "name", f"{prefix}.name")
        if not SAFE_PLUGIN_NAME.fullmatch(name):
            raise ConfigError(
                f'feature plugin name "{name}" is unsafe; use only letters, numbers, '
                "underscore, and hyphen"
            )
        if name in plugin_names:
            raise ConfigError(f'duplicate feature plugin name "{name}"')
        plugin_names.add(name)
        entrypoint = _non_empty_string(
            plugin_raw, "entrypoint", f"{prefix}.entrypoint"
        )
        params = plugin_raw.get("params", {})
        if not isinstance(params, dict) or not all(
            isinstance(key, str) for key in params
        ):
            raise ConfigError(f'"{prefix}.params" must be a mapping with string keys')
        try:
            json.dumps(params)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f'"{prefix}.params" values must be JSON-serializable') from exc
        plugins.append(
            FeaturePluginConfig(
                name=name,
                entrypoint=entrypoint,
                params=dict(params),
            )
        )

    sets_raw = features_raw.get("sets")
    if not isinstance(sets_raw, list) or not sets_raw:
        raise ConfigError('"features.sets" must be a non-empty list')
    feature_sets: list[FeatureSetConfig] = []
    set_names: set[str] = set()
    for index, set_raw in enumerate(sets_raw):
        prefix = f"features.sets[{index}]"
        if not isinstance(set_raw, dict):
            raise ConfigError(f'"{prefix}" must be a mapping')
        _reject_unknown(set_raw, {"name", "source_columns", "plugins"}, prefix)
        name = _non_empty_string(set_raw, "name", f"{prefix}.name")
        if not SAFE_FEATURE_SET_NAME.fullmatch(name):
            raise ConfigError(
                f'feature-set name "{name}" is unsafe; use only letters, numbers, underscore, and hyphen'
            )
        if name in set_names:
            raise ConfigError(f'duplicate feature-set name "{name}"')
        set_names.add(name)

        source_columns = set_raw.get("source_columns")
        if not isinstance(source_columns, list) or not source_columns:
            raise ConfigError(f'"{prefix}.source_columns" must be a non-empty list')
        if not all(isinstance(column, str) and column for column in source_columns):
            raise ConfigError(
                f'"{prefix}.source_columns" must contain non-empty strings'
            )
        if len(source_columns) != len(set(source_columns)):
            raise ConfigError(f'"{prefix}.source_columns" must contain unique values')
        if "*" in source_columns and source_columns != ["*"]:
            raise ConfigError(f'"{prefix}.source_columns": "*" must appear alone')

        referenced_plugins = set_raw.get("plugins", [])
        if not isinstance(referenced_plugins, list) or not all(
            isinstance(plugin_name, str) and plugin_name
            for plugin_name in referenced_plugins
        ):
            raise ConfigError(f'"{prefix}.plugins" must be a list of non-empty strings')
        if len(referenced_plugins) != len(set(referenced_plugins)):
            raise ConfigError(f'"{prefix}.plugins" must contain unique plugin names')
        unknown = [
            plugin_name
            for plugin_name in referenced_plugins
            if plugin_name not in plugin_names
        ]
        if unknown:
            raise ConfigError(
                f'feature set "{name}" references unknown plugin "{unknown[0]}"'
            )
        feature_sets.append(
            FeatureSetConfig(
                name=name,
                source_columns=list(source_columns),
                plugins=list(referenced_plugins),
            )
        )
    return FeaturesConfig(plugins=plugins, sets=feature_sets)


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

    _reject_unknown(
        raw,
        {"schema_version", "project", "task", "data", "validation", "split",
         "preprocessing", "features", "models", "evaluation", "training", "hpo"},
        "top-level",
    )
    schema_version = raw.get("schema_version")
    if schema_version != "0.1":
        raise ConfigError('"schema_version" must be the string "0.1"')

    project_raw = _mapping(raw, "project")
    task_raw = _mapping(raw, "task")
    data_raw = _mapping(raw, "data")
    _reject_unknown(project_raw, {"name"}, "project")
    _reject_unknown(task_raw, {"type", "target", "positive_class"}, "task")
    _reject_unknown(data_raw, {"path", "format"}, "data")

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
    _reject_unknown(validation_raw, {"enabled", "fail_on_error"}, "validation")
    validation = ValidationConfig(
        enabled=_boolean(validation_raw, "enabled", True, "validation.enabled"),
        fail_on_error=_boolean(
            validation_raw, "fail_on_error", True, "validation.fail_on_error"
        ),
    )

    split_raw = raw.get("split", {})
    if not isinstance(split_raw, dict):
        raise ConfigError('configuration section "split" must be a mapping')
    _reject_unknown(
        split_raw, {"validation_ratio", "test_ratio", "stratify", "random_seed"}, "split"
    )
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
    _reject_unknown(preprocessing_raw, {"external"}, "preprocessing")
    external_raw = preprocessing_raw.get("external", {})
    if not isinstance(external_raw, dict):
        raise ConfigError('"preprocessing.external" must be a mapping')
    _reject_unknown(
        external_raw, {"enabled", "entrypoint", "params"}, "preprocessing.external"
    )
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
    features = _feature_config(raw)
    models = _model_config(raw)
    evaluation = _evaluation_config(raw, task_type)
    training = _training_config(raw)
    hpo = _hpo_config(raw)

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
        features=features,
        models=models,
        evaluation=evaluation,
        training=training,
        config_path=config_path,
        hpo=hpo,
    )
