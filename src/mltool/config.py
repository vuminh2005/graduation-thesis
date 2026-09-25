"""Load and validate the intentionally small Phase-1 through Phase-5 schema."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import yaml


SUPPORTED_TASKS = {"binary", "multiclass", "regression"}
SUPPORTED_FORMATS = {"auto", "csv", "parquet"}
# The five isolated single-family candidates from Phase 4 onward.
SINGLE_MODEL_FAMILIES = {"GBM", "CAT", "XGB", "RF", "XT"}
# Phase 8 adds ENSEMBLE: several of the families above, bagged/stacked/weighted
# by AutoGluon itself, as one more candidate in the FeatureSet x model matrix.
# Phase 11 adds SKLEARN: a user's sklearn-compatible estimator (never an ENSEMBLE member).
SUPPORTED_MODEL_FAMILIES = SINGLE_MODEL_FAMILIES | {"ENSEMBLE", "SKLEARN"}
SKLEARN_INPUT_MODES = ("auto", "raw")
# families defaults to all five single families, in this documented order.
DEFAULT_ENSEMBLE_FAMILIES = ["GBM", "CAT", "XGB", "RF", "XT"]
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
# Each family's own seed hyperparameter (AutoGluon's AbstractModel.seed_name).
FAMILY_SEED_KEYS = {
    "GBM": "seed",
    "CAT": "random_seed",
    "XGB": "seed",
    "RF": "random_state",
    "XT": "random_state",
    # SKLEARN: only when the user's estimator has a top-level random_state.
    "SKLEARN": "random_state",
}
# Local (Ray-free) searchers of the installed AutoGluon; "grid" is its local_grid.
SUPPORTED_SEARCHERS = ("random", "grid")
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
    # Row identifiers `mltool score` copies into its output; nothing else uses them.
    id_columns: list[str] = field(default_factory=list)


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
    # Columns the plugins may read but that are not features themselves, so a
    # plugin can replace a column instead of only adding to it (Phase 9).
    plugin_inputs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeaturesConfig:
    plugins: list[FeaturePluginConfig]
    sets: list[FeatureSetConfig]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)
    # Normalized specs, e.g. {"learning_rate": {"type": "real", "low": 0.005,
    # "high": 0.2, "log": True}}; used by ``tune`` only, never by ``train``.
    search_space: dict[str, dict[str, Any]] = field(default_factory=dict)
    # SKLEARN only: "<file>:<attribute>" as written, the resolved file, the input mode.
    entrypoint: str | None = None
    source_path: Path | None = None
    input: str | None = None


def source_sha256(path: Path | None) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None
    except OSError:
        return None


def execute_source(module: Any, path: Path) -> None:
    """Run a user's file as ``module`` from the bytes it holds now.

    ``SourceFileLoader.exec_module`` would reuse ``__pycache__/*.pyc``, which it
    validates only by the source's size and mtime in whole seconds: an edit that
    keeps the size, within the same second, would run the old code while
    ``source_sha256`` hashes the new file. Compiling the source directly also
    leaves no ``__pycache__`` in the user's project.
    """
    code = compile(Path(path).read_bytes(), str(path), "exec", dont_inherit=True)
    exec(code, module.__dict__)


def relative_path(path: Path | str, start: Path | str) -> str:
    """``path`` as recorded in a manifest: relative to the directory holding that manifest.

    Recorded paths stay valid when a project or a registry version is moved or copied.
    """
    return Path(os.path.relpath(Path(path), Path(start))).as_posix()


def model_record(model: ModelConfig) -> dict[str, Any]:
    """A model as persisted in every artifact and compared by every staleness check.

    For SKLEARN it also carries the entrypoint, the input mode and the hash of the
    entrypoint file, so editing the user's model file makes the artifacts stale.
    Other families keep exactly the three keys they always had.
    """
    record: dict[str, Any] = {"name": model.name, "family": model.family, "params": model.params}
    if model.family == "SKLEARN":
        record.update(
            entrypoint=model.entrypoint,
            input=model.input,
            source_sha256=source_sha256(model.source_path),
        )
    return record


@dataclass(frozen=True)
class CvConfig:
    """Cross-validated candidate evaluation (Phase 9); absent means single holdout."""

    folds: int
    repeats: int

    @property
    def total_fits(self) -> int:
        return self.folds * self.repeats


@dataclass(frozen=True)
class EvaluationConfig:
    primary_metric: str
    secondary_metrics: list[str]
    cv: CvConfig | None = None

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
    # null = detect from the process's cgroup (see mltool.resources)
    memory_limit_gb: float | None = None
    num_cpus: int | None = None


@dataclass(frozen=True)
class HpoConfig:
    top_n: int
    num_trials: int
    time_limit_seconds: int
    searcher: str = "random"


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


def _ensemble_params(params_raw: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Validate and normalize an ``ENSEMBLE`` model's params, defaults filled in.

    Kept as a plain JSON-serializable dict in ``ModelConfig.params`` (like every
    other family) rather than a new dataclass field, so every place that already
    forwards ``model.params`` verbatim (training/tuning/finalize results,
    ``selected.json``, the registry, MLflow params) exposes the ensemble
    configuration for free.
    """
    _reject_unknown(params_raw, {"families", "num_bag_folds", "num_stack_levels"}, f"{prefix}.params")

    families = params_raw.get("families", list(DEFAULT_ENSEMBLE_FAMILIES))
    if not isinstance(families, list) or not families:
        raise ConfigError(f'"{prefix}.params.families" must be a non-empty list')
    if not all(isinstance(family, str) for family in families):
        raise ConfigError(f'"{prefix}.params.families" must be a list of strings')
    if len(families) != len(set(families)):
        raise ConfigError(f'"{prefix}.params.families" must contain unique values')
    if "SKLEARN" in families:
        raise ConfigError(
            f'"{prefix}.params.families" contains SKLEARN; custom SKLEARN models cannot be '
            "ENSEMBLE members"
        )
    unsupported = [family for family in families if family not in SINGLE_MODEL_FAMILIES]
    if unsupported:
        supported = ", ".join(sorted(SINGLE_MODEL_FAMILIES))
        raise ConfigError(
            f'"{prefix}.params.families" contains unsupported family "{unsupported[0]}" '
            f'(ENSEMBLE may not nest itself); expected a subset of: {supported}'
        )

    num_bag_folds = params_raw.get("num_bag_folds", 3)
    if isinstance(num_bag_folds, bool) or not isinstance(num_bag_folds, int) or (
        num_bag_folds != 0 and num_bag_folds < 2
    ):
        raise ConfigError(f'"{prefix}.params.num_bag_folds" must be 0 or an integer >= 2')

    num_stack_levels = params_raw.get("num_stack_levels", 1)
    if isinstance(num_stack_levels, bool) or not isinstance(num_stack_levels, int) or (
        num_stack_levels < 0
    ):
        raise ConfigError(f'"{prefix}.params.num_stack_levels" must be a non-negative integer')
    if num_stack_levels > 0 and num_bag_folds < 2:
        raise ConfigError(
            f'"{prefix}.params.num_stack_levels" greater than 0 requires '
            f'"{prefix}.params.num_bag_folds" to be at least 2'
        )

    return {
        "families": list(families),
        "num_bag_folds": num_bag_folds,
        "num_stack_levels": num_stack_levels,
    }


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(
        value
    )


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _search_space_entry(spec: Any, where: str) -> dict[str, Any]:
    """One declared hyperparameter range, normalized to a canonical JSON form."""
    if not isinstance(spec, dict):
        raise ConfigError(f'"{where}" must be a mapping')
    kind = spec.get("type")
    if kind == "real":
        _reject_unknown(spec, {"type", "low", "high", "log", "default"}, where)
        low, high, log = spec.get("low"), spec.get("high"), spec.get("log", False)
        if not _number(low) or not _number(high):
            raise ConfigError(f'"{where}.low" and "{where}.high" must be numbers')
        if not isinstance(log, bool):
            raise ConfigError(f'"{where}.log" must be true or false')
        entry: dict[str, Any] = {"type": "real", "low": float(low), "high": float(high), "log": log}
    elif kind == "int":
        # autogluon.common.space.Int(lower, upper, default) has no log scale.
        _reject_unknown(spec, {"type", "low", "high", "default"}, where)
        low, high = spec.get("low"), spec.get("high")
        if not _integer(low) or not _integer(high):
            raise ConfigError(f'"{where}.low" and "{where}.high" must be integers')
        entry = {"type": "int", "low": low, "high": high}
    elif kind == "categorical":
        _reject_unknown(spec, {"type", "values", "default"}, where)
        values = spec.get("values")
        if not isinstance(values, list) or not values:
            raise ConfigError(f'"{where}.values" must be a non-empty list')
        if not all(v is None or isinstance(v, (str, bool, int)) or _number(v) for v in values):
            raise ConfigError(f'"{where}.values" must contain only JSON scalars')
        # JSON identity, so true and 1 stay distinct
        encoded = [json.dumps(value) for value in values]
        if len(set(encoded)) != len(encoded):
            raise ConfigError(f'"{where}.values" must contain unique values')
        entry = {"type": "categorical", "values": list(values)}
        if "default" in spec:
            if json.dumps(spec["default"]) not in encoded:
                raise ConfigError(f'"{where}.default" must be one of "{where}.values"')
            entry["default"] = spec["default"]
        return entry
    else:
        raise ConfigError(f'"{where}.type" must be one of: real, int, categorical')

    if not entry["low"] < entry["high"]:
        raise ConfigError(f'"{where}.low" must be less than "{where}.high"')
    if entry.get("log") and entry["low"] <= 0:
        raise ConfigError(f'"{where}.log" requires "{where}.low" greater than 0')
    if "default" in spec:
        default = spec["default"]
        valid = _integer(default) if kind == "int" else _number(default)
        if not valid or not entry["low"] <= default <= entry["high"]:
            raise ConfigError(
                f'"{where}.default" must be a {"integer" if kind == "int" else "number"} '
                f'within [low, high]'
            )
        entry["default"] = default if kind == "int" else float(default)
    return entry


def _search_space(
    raw: Any, prefix: str, family: str, params: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    where = f"{prefix}.search_space"
    if raw is None:
        return {}
    if family == "ENSEMBLE":
        raise ConfigError(
            f'"{where}" is not supported for ENSEMBLE; HPO is excluded for ENSEMBLE by design'
        )
    if not isinstance(raw, dict) or not all(isinstance(key, str) and key for key in raw):
        raise ConfigError(f'"{where}" must be a mapping with non-empty string keys')
    overlap = sorted(set(raw) & set(params))
    if overlap:
        raise ConfigError(
            f'"{where}" and "{prefix}.params" both set: {", ".join(overlap)}; '
            "a hyperparameter is either fixed or searched"
        )
    seed_key = FAMILY_SEED_KEYS[family]
    if seed_key in raw:
        raise ConfigError(
            f'"{where}.{seed_key}" is the {family} seed, set by training.seed or params; '
            "it cannot be searched"
        )
    return {key: _search_space_entry(spec, f"{where}.{key}") for key, spec in raw.items()}


def _sklearn_entrypoint(model_raw: dict[str, Any], prefix: str, config_dir: Path) -> tuple[str, Path]:
    entrypoint = model_raw.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ConfigError(f'"{prefix}.entrypoint" is required for SKLEARN: "<python-file>:<name>"')
    file_text, separator, attribute = entrypoint.strip().rpartition(":")
    if not separator or not file_text.strip() or not attribute.strip().isidentifier():
        raise ConfigError(f'"{prefix}.entrypoint" must be "<python-file>:<class-or-function>"')
    path = Path(file_text.strip()).expanduser()
    path = (path if path.is_absolute() else config_dir / path).resolve()
    if not path.is_file():
        raise ConfigError(f'"{prefix}.entrypoint" file not found: {path}')
    return entrypoint.strip(), path


def _model_config(raw: dict[str, Any], config_dir: Path) -> list[ModelConfig]:
    models_raw = raw.get("models", [])
    if not isinstance(models_raw, list):
        raise ConfigError('"models" must be a list')
    models: list[ModelConfig] = []
    names: set[str] = set()
    for index, model_raw in enumerate(models_raw):
        prefix = f"models[{index}]"
        if not isinstance(model_raw, dict):
            raise ConfigError(f'"{prefix}" must be a mapping')
        allowed = {"name", "family", "params", "search_space"}
        if model_raw.get("family") == "SKLEARN":
            allowed |= {"entrypoint", "input"}
        _reject_unknown(model_raw, allowed, prefix)
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
        params_raw = model_raw.get("params", {})
        if not isinstance(params_raw, dict) or not all(isinstance(key, str) for key in params_raw):
            raise ConfigError(f'"{prefix}.params" must be a mapping with string keys')
        if family == "ENSEMBLE":
            params = _ensemble_params(params_raw, prefix)
        else:
            params = _json_mapping(params_raw, f"{prefix}.params")
        search_space = _search_space(model_raw.get("search_space"), prefix, family, params)
        entrypoint = source_path = input_mode = None
        if family == "SKLEARN":
            entrypoint, source_path = _sklearn_entrypoint(model_raw, prefix, config_dir)
            input_mode = model_raw.get("input", "auto")
            if input_mode not in SKLEARN_INPUT_MODES:
                raise ConfigError(f'"{prefix}.input" must be one of: {", ".join(SKLEARN_INPUT_MODES)}')
            reserved = sorted(k for k in [*params, *search_space] if k.startswith("mltool_"))
            if reserved:
                raise ConfigError(f'"{prefix}": keys starting with "mltool_" are reserved: {reserved}')
        models.append(
            ModelConfig(
                name=name, family=family, params=params, search_space=search_space,
                entrypoint=entrypoint, source_path=source_path, input=input_mode,
            )
        )
    return models


def _evaluation_config(raw: dict[str, Any], task_type: str) -> EvaluationConfig:
    default_primary, default_secondary = DEFAULT_METRICS[task_type]
    evaluation_raw = raw.get("evaluation", {})
    if not isinstance(evaluation_raw, dict):
        raise ConfigError('configuration section "evaluation" must be a mapping')
    _reject_unknown(evaluation_raw, {"primary_metric", "secondary_metrics", "cv"}, "evaluation")
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
    return EvaluationConfig(
        primary_metric=primary, secondary_metrics=secondary, cv=_cv_config(evaluation_raw)
    )


def _cv_config(evaluation_raw: dict[str, Any]) -> CvConfig | None:
    """``evaluation.cv``; absent (or null) keeps the single-holdout behavior."""
    if "cv" not in evaluation_raw or evaluation_raw["cv"] is None:
        return None
    cv_raw = evaluation_raw["cv"]
    if not isinstance(cv_raw, dict):
        raise ConfigError('configuration section "evaluation.cv" must be a mapping')
    _reject_unknown(cv_raw, {"folds", "repeats"}, "evaluation.cv")
    folds = cv_raw.get("folds", 5)
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ConfigError('"evaluation.cv.folds" must be an integer of at least 2')
    repeats = cv_raw.get("repeats", 1)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ConfigError('"evaluation.cv.repeats" must be an integer of at least 1')
    return CvConfig(folds=folds, repeats=repeats)


def _training_config(raw: dict[str, Any]) -> TrainingConfig:
    training_raw = raw.get("training", {})
    if not isinstance(training_raw, dict):
        raise ConfigError('configuration section "training" must be a mapping')
    _reject_unknown(
        training_raw,
        {"time_limit_seconds", "seed", "memory_limit_gb", "num_cpus"},
        "training",
    )
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
    memory_limit_gb = training_raw.get("memory_limit_gb")
    if memory_limit_gb is not None and (
        isinstance(memory_limit_gb, bool)
        or not isinstance(memory_limit_gb, (int, float))
        or not math.isfinite(memory_limit_gb)
        or memory_limit_gb <= 0
    ):
        raise ConfigError('"training.memory_limit_gb" must be null or a positive number')
    num_cpus = training_raw.get("num_cpus")
    if num_cpus is not None and (
        isinstance(num_cpus, bool) or not isinstance(num_cpus, int) or num_cpus <= 0
    ):
        raise ConfigError('"training.num_cpus" must be null or a positive integer')
    return TrainingConfig(
        time_limit_seconds=time_limit,
        seed=seed,
        memory_limit_gb=None if memory_limit_gb is None else float(memory_limit_gb),
        num_cpus=num_cpus,
    )


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
    _reject_unknown(hpo_raw, {"top_n", "num_trials", "time_limit_seconds", "searcher"}, "hpo")
    searcher = hpo_raw.get("searcher", "random")
    if searcher not in SUPPORTED_SEARCHERS:
        raise ConfigError(f'"hpo.searcher" must be one of: {", ".join(SUPPORTED_SEARCHERS)}')
    return HpoConfig(
        top_n=_positive_int(hpo_raw, "top_n", 3),
        num_trials=_positive_int(hpo_raw, "num_trials", 10),
        time_limit_seconds=_positive_int(hpo_raw, "time_limit_seconds", None),
        searcher=searcher,
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
        _reject_unknown(
            set_raw, {"name", "source_columns", "plugins", "plugin_inputs"}, prefix
        )
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
        plugin_inputs = set_raw.get("plugin_inputs", [])
        if not isinstance(plugin_inputs, list) or not all(
            isinstance(column, str) and column for column in plugin_inputs
        ):
            raise ConfigError(f'"{prefix}.plugin_inputs" must be a list of non-empty strings')
        if len(plugin_inputs) != len(set(plugin_inputs)):
            raise ConfigError(f'"{prefix}.plugin_inputs" must contain unique values')
        if "*" in plugin_inputs:
            raise ConfigError(
                f'"{prefix}.plugin_inputs" may not contain "*"; list the columns explicitly'
            )
        if plugin_inputs and not referenced_plugins:
            raise ConfigError(
                f'"{prefix}.plugin_inputs" requires at least one plugin in '
                f'"{prefix}.plugins"; the columns would otherwise be read by nothing'
            )
        feature_sets.append(
            FeatureSetConfig(
                name=name,
                source_columns=list(source_columns),
                plugins=list(referenced_plugins),
                plugin_inputs=list(plugin_inputs),
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
    _reject_unknown(data_raw, {"path", "format", "id_columns"}, "data")

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
    id_columns = data_raw.get("id_columns", [])
    if not isinstance(id_columns, list) or not all(
        isinstance(column, str) and column for column in id_columns
    ):
        raise ConfigError('"data.id_columns" must be a list of non-empty strings')
    if len(id_columns) != len(set(id_columns)):
        raise ConfigError('"data.id_columns" must contain unique values')
    if target in id_columns:
        raise ConfigError(f'"data.id_columns" may not contain the target column "{target}"')

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
    models = _model_config(raw, config_path.parent)
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
        data=DataConfig(path=data_path, format=data_format, id_columns=list(id_columns)),
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
