"""The only Phase-4 module that directly calls AutoGluon APIs."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from mltool.config import HpoConfig, ModelConfig, TaskConfig, TrainingConfig


AUTOGLUON_METRICS = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "log_loss": "log_loss",
    "rmse": "root_mean_squared_error",
    "mae": "mean_absolute_error",
    "r2": "r2",
}

FAMILY_MODEL_TOKENS = {
    "GBM": ("LightGBM",),
    "CAT": ("CatBoost",),
    "XGB": ("XGBoost",),
    "RF": ("RandomForest",),
    "XT": ("ExtraTrees",),
}


# AutoGluon 1.6 has no seed argument on TabularPredictor.fit. Reproducibility is
# controlled by (a) the learner's random_state (internal holdout split), set via
# TabularPredictor(learner_kwargs=...), and (b) each model's own seed
# hyperparameter, whose name is family specific (AbstractModel.seed_name).
FAMILY_SEED_KEYS = {
    "GBM": "seed",
    "CAT": "random_seed",
    "XGB": "seed",
    "RF": "random_state",
    "XT": "random_state",
}

# Families for which AutoGluon defines no default hyperparameter search space:
# with hyperparameter_tune_kwargs they train a single default model.
NO_SEARCH_SPACE_FAMILIES = frozenset({"RF", "XT"})


class AutoGluonError(RuntimeError):
    """An expected candidate-level AutoGluon failure."""


@dataclass
class AutoGluonOutput:
    predictions: pd.Series
    probabilities: pd.DataFrame | None
    positive_class: Any | None
    trained_models: list[str]
    autogluon_version: str
    best_model: str | None = None
    best_hyperparameters: dict[str, Any] | None = None


def _best_hyperparameters(predictor: Any, model_name: str) -> dict[str, Any]:
    info = predictor.info()["model_info"][model_name]
    return dict(info.get("hyperparameters", {}))


class AutoGluonAdapter:
    """Fit exactly one configured model family on train and predict validation."""

    def __init__(
        self,
        predictor_factory: Callable[..., Any] | None = None,
        version_resolver: Callable[[], str] | None = None,
    ) -> None:
        self._predictor_factory = predictor_factory
        self._version_resolver = version_resolver or (
            lambda: version("autogluon.tabular")
        )

    def _factory(self) -> Callable[..., Any]:
        if self._predictor_factory is None:
            try:
                from autogluon.tabular import TabularPredictor
            except Exception as exc:  # pragma: no cover - packaging failure path
                raise AutoGluonError(f"AutoGluon Tabular is unavailable: {exc}") from exc
            self._predictor_factory = TabularPredictor
        return self._predictor_factory

    def fit_predict(
        self,
        *,
        train_data: pd.DataFrame,
        validation_features: pd.DataFrame,
        task: TaskConfig,
        model: ModelConfig,
        primary_metric: str,
        predictor_path: Path,
        training: TrainingConfig,
        hpo: HpoConfig | None = None,
    ) -> AutoGluonOutput:
        predictor_kwargs: dict[str, Any] = {
            "label": task.target,
            "problem_type": task.type,
            "eval_metric": AUTOGLUON_METRICS[primary_metric],
            "path": str(predictor_path),
            "verbosity": 2,
        }
        if task.type == "binary" and task.positive_class is not None:
            predictor_kwargs["positive_class"] = task.positive_class
        model_params = dict(model.params)
        if training.seed is not None:
            predictor_kwargs["learner_kwargs"] = {"random_state": training.seed}
            # An explicitly fixed model seed in the config wins over training.seed.
            model_params.setdefault(FAMILY_SEED_KEYS[model.family], training.seed)

        fit_kwargs: dict[str, Any] = {
            "train_data": train_data.copy(deep=True),
            "hyperparameters": {model.family: model_params},
            "hyperparameter_tune_kwargs": None,
            "fit_weighted_ensemble": False,
            "fit_full_last_level_weighted_ensemble": False,
            "full_weighted_ensemble_additionally": False,
            "num_bag_folds": 0,
            "num_stack_levels": 0,
            "dynamic_stacking": False,
            "num_gpus": 0,
            "fit_strategy": "sequential",
        }
        if hpo is not None:
            # Phase 5: local, sequential random search bounded by trials and time.
            # Everything else (no bagging/stacking/ensemble/GPU) is unchanged.
            fit_kwargs["hyperparameter_tune_kwargs"] = {
                "num_trials": hpo.num_trials,
                "scheduler": "local",
                "searcher": "random",
            }
            fit_kwargs["time_limit"] = hpo.time_limit_seconds
        elif training.time_limit_seconds is not None:
            fit_kwargs["time_limit"] = training.time_limit_seconds

        try:
            predictor = self._factory()(**predictor_kwargs)
            # Deliberately no tuning_data: MLTool's validation split remains
            # external and is used only after fitting for cross-candidate metrics.
            predictor.fit(**fit_kwargs)
            predictions = predictor.predict(validation_features.copy(deep=True))
            probabilities = None
            if task.type in {"binary", "multiclass"}:
                probabilities = predictor.predict_proba(
                    validation_features.copy(deep=True),
                    as_multiclass=True,
                )
            trained_models = list(predictor.model_names())
            best_model = best_hyperparameters = None
            if hpo is not None:
                best_model = str(predictor.model_best)
                best_hyperparameters = _best_hyperparameters(predictor, best_model)
        except (Exception, SystemExit) as exc:
            raise AutoGluonError(str(exc)) from exc

        if not isinstance(predictions, pd.Series):
            predictions = pd.Series(predictions, index=validation_features.index)
        if probabilities is not None and not isinstance(probabilities, pd.DataFrame):
            probabilities = pd.DataFrame(probabilities, index=validation_features.index)
        if len(predictions) != len(validation_features):
            raise AutoGluonError("AutoGluon returned the wrong number of validation predictions")
        if probabilities is not None and len(probabilities) != len(validation_features):
            raise AutoGluonError("AutoGluon returned the wrong number of probability rows")
        if not trained_models:
            raise AutoGluonError("AutoGluon did not train a usable model")
        if any("WeightedEnsemble" in name for name in trained_models):
            raise AutoGluonError("AutoGluon unexpectedly trained a weighted ensemble")
        expected_tokens = FAMILY_MODEL_TOKENS[model.family]
        unrelated = [
            name for name in trained_models if not any(token in name for token in expected_tokens)
        ]
        if unrelated:
            raise AutoGluonError(
                "AutoGluon unexpectedly trained models outside family "
                f'{model.family}: {", ".join(unrelated)}'
            )

        resolved_positive = None
        if task.type == "binary":
            resolved_positive = getattr(predictor, "positive_class", task.positive_class)
        return AutoGluonOutput(
            predictions=predictions.copy(deep=True),
            probabilities=probabilities.copy(deep=True) if probabilities is not None else None,
            positive_class=resolved_positive,
            trained_models=trained_models,
            autogluon_version=self._version_resolver(),
            best_model=best_model,
            best_hyperparameters=best_hyperparameters,
        )
