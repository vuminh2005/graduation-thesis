"""User-supplied sklearn-compatible estimators (``models[].family: SKLEARN``).

Nothing here imports AutoGluon; the AbstractModel wrapper that AutoGluon trains
lives in ``mltool.autogluon_sklearn``. This module resolves and loads the user's
entrypoint, validates the object it builds, decides the seed, and implements
``input: auto`` preprocessing.

Loading: the entrypoint file is executed as a module under a private name that
is removed from ``sys.modules`` again (as for feature plugins). cloudpickle
serializes objects whose module cannot be imported *by value*, so a fitted
estimator built from user code is saved together with that code and loads in a
process that never reads the user's file.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd

from mltool.config import ConfigError, ModelConfig, TrainingConfig, execute_source, source_sha256

SKLEARN = "SKLEARN"
# Hyperparameter keys MLTool adds for the wrapper; never passed to the estimator.
PRIVATE_PREFIX = "mltool_"
ENTRYPOINT_KEY = "mltool_entrypoint"
INPUT_KEY = "mltool_input"
RANDOM_STATE_KEY = "mltool_random_state"


class CustomModelError(ConfigError):
    """A user model entrypoint that cannot be loaded or does not meet the contract."""


def entrypoint_spec(model: ModelConfig) -> str:
    """``<absolute file>:<attribute>``: what the wrapper loads at fit time."""
    return f"{model.source_path}:{model.entrypoint.rpartition(':')[2]}"


def load_entrypoint(spec: str) -> Callable[..., Any]:
    """Execute the entrypoint file and return the named callable (class or function)."""
    file_text, _, attribute = spec.rpartition(":")
    path = Path(file_text)
    if not path.is_file():
        raise CustomModelError(f"model entrypoint file not found: {path}")
    digest = hashlib.sha256(f"{path.resolve()}:{attribute}".encode()).hexdigest()[:16]
    module_name = f"_mltool_user_model_{digest}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise CustomModelError(f"could not create a module loader for {path}")
    module = importlib.util.module_from_spec(module_spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module  # visible while it executes (dataclasses etc.)
    try:
        execute_source(module, path)
    except (Exception, SystemExit) as exc:
        raise CustomModelError(f"could not load model entrypoint {path}: {exc}") from exc
    finally:
        # Not importable afterwards, so cloudpickle stores its objects by value.
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    target = getattr(module, attribute, None)
    if target is None:
        raise CustomModelError(f'"{attribute}" was not found in {path}')
    if not callable(target):
        raise CustomModelError(f'"{attribute}" in {path} is not a class or function')
    return target


def estimator_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if not k.startswith(PRIVATE_PREFIX)}


def build_estimator(factory: Callable[..., Any], params: dict[str, Any]) -> Any:
    try:
        return factory(**params)
    except Exception as exc:
        raise CustomModelError(f"calling the model entrypoint with {sorted(params)} failed: {exc}") from exc


def search_space_defaults(search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The value AutoGluon's first trial uses for each declared range."""
    values = {}
    for key, spec in search_space.items():
        if "default" in spec:
            values[key] = spec["default"]
        elif spec["type"] == "categorical":
            values[key] = spec["values"][0]
        else:
            values[key] = spec["low"]
    return values


@dataclass(frozen=True)
class Probe:
    has_random_state: bool
    methods: frozenset[str]


@lru_cache(maxsize=64)
def _probe(spec: str, sha: str | None, params_json: str) -> Probe:
    estimator = build_estimator(load_entrypoint(spec), json.loads(params_json))
    methods = frozenset(
        name for name in ("fit", "predict", "predict_proba") if callable(getattr(estimator, name, None))
    )
    get_params = getattr(estimator, "get_params", None)
    has_random_state = False
    if callable(get_params):
        try:
            has_random_state = "random_state" in get_params(deep=False)
        except Exception:
            has_random_state = False
    return Probe(has_random_state=has_random_state, methods=methods)


def probe(model: ModelConfig) -> Probe:
    """Build one instance (params plus each search range's first value) and inspect it."""
    params = {**model.params, **search_space_defaults(model.search_space)}
    return _probe(entrypoint_spec(model), source_sha256(model.source_path),
                  json.dumps(params, sort_keys=True, default=str))


def validate(model: ModelConfig, task_type: str) -> Probe:
    """The contract, checked on a real instance: fit/predict, predict_proba to classify."""
    result = probe(model)
    required = ["fit", "predict"] + (["predict_proba"] if task_type != "regression" else [])
    missing = [name for name in required if name not in result.methods]
    if missing:
        hint = ""
        if "predict_proba" in missing:
            hint = (" (for sklearn's SVC set probability: true; classification needs "
                    "class probabilities)")
        raise CustomModelError(
            f'model "{model.name}": the object built by {model.entrypoint} has no '
            f"{', '.join(f'{name}()' for name in missing)}{hint}"
        )
    return result


def seed_decision(model: ModelConfig, training: TrainingConfig) -> tuple[Any | None, str]:
    """The estimator's effective seed and why.

    Only a top-level ``random_state`` is managed; estimators nested inside the
    user's object (e.g. AdaBoost's base estimator) are the user's responsibility.
    """
    if "random_state" in model.params:
        return model.params["random_state"], "random_state fixed in params"
    if not probe(model).has_random_state:
        return None, "estimator has no random_state parameter; its randomness is not seeded by MLTool"
    if training.seed is None:
        return None, "training.seed is null"
    return training.seed, "random_state set from training.seed"


def seed_note_record(model: ModelConfig, training: TrainingConfig) -> dict[str, str]:
    """``{"seed_note": ...}`` for a SKLEARN model, nothing for other families."""
    return {"seed_note": seed_decision(model, training)[1]} if model.family == SKLEARN else {}


def align_probabilities(proba: Any, classes: Any, n_classes: int) -> np.ndarray:
    """Columns ordered 0..n_classes-1, as AutoGluon's label-encoded classes.

    A class absent from the fit rows gets a zero column instead of shifting the
    others.
    """
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    if classes is None:
        if proba.shape[1] != n_classes:
            raise ValueError(f"predict_proba returned {proba.shape[1]} columns for {n_classes} classes")
        return proba
    full = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    for column, label in enumerate(np.asarray(classes).tolist()):
        index = int(label)
        if not 0 <= index < n_classes:
            raise ValueError(f"estimator class {label!r} is outside 0..{n_classes - 1}")
        full[:, index] = proba[:, column]
    return full


class AutoInput:
    """``input: auto``: statistics learned only from the rows ``fit`` sees.

    Numeric columns: median imputation, then standard scaling. Categorical
    columns: most-frequent imputation, then one-hot with the categories seen in
    fit (a category unseen in fit becomes an all-zero row). Returns float64.
    """

    def fit(self, frame: pd.DataFrame) -> "AutoInput":
        self.columns_ = list(frame.columns)
        self.numeric_ = [c for c in frame.columns if _is_numeric(frame[c])]
        self.categorical_ = [c for c in frame.columns if c not in self.numeric_]
        self.medians_: dict[str, float] = {}
        self.means_: dict[str, float] = {}
        self.stds_: dict[str, float] = {}
        for column in self.numeric_:
            values = pd.to_numeric(frame[column], errors="coerce").astype(float)
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            std = float(filled.std(ddof=0))
            self.medians_[column] = median
            self.means_[column] = float(filled.mean())
            self.stds_[column] = std if std > 0 else 1.0
        self.modes_: dict[str, Any] = {}
        self.categories_: dict[str, list[Any]] = {}
        for column in self.categorical_:
            values = frame[column].astype(object)
            present = values.dropna()
            self.modes_[column] = present.mode().iloc[0] if len(present) else None
            filled = values.where(values.notna(), self.modes_[column])
            self.categories_[column] = sorted(filled.dropna().unique().tolist(), key=repr)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts = []
        for column in self.numeric_:
            values = pd.to_numeric(frame[column], errors="coerce").astype(float)
            values = values.fillna(self.medians_[column])
            parts.append(((values - self.means_[column]) / self.stds_[column]).to_numpy()[:, None])
        for column in self.categorical_:
            values = frame[column].astype(object)
            values = values.where(values.notna(), self.modes_[column])
            for category in self.categories_[column]:
                parts.append((values == category).to_numpy(dtype=np.float64)[:, None])
        if not parts:
            return np.zeros((len(frame), 0))
        return np.hstack(parts).astype(np.float64)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not isinstance(series.dtype, pd.CategoricalDtype)
