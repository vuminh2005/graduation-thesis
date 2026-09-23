"""The only Phase-4 module that directly calls AutoGluon APIs."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import re
from typing import Any, Callable

import pandas as pd

from mltool.config import HpoConfig, ModelConfig, TaskConfig, TrainingConfig
from mltool.resources import resolve_resource_limits


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


def effective_model_seed(model: ModelConfig, training: TrainingConfig) -> Any | None:
    """The seed the model's own hyperparameters receive in ``fit``.

    A seed fixed in the model's ``params`` wins over ``training.seed``. An
    ENSEMBLE candidate has no single family seed key of its own: the same
    ``training.seed`` value is applied to every member family's seed key
    (Step 4), so it is also what "effective seed" means for the candidate as
    a whole.
    """
    if model.family == "ENSEMBLE":
        return training.seed
    params = dict(model.params)
    if training.seed is not None:
        params.setdefault(FAMILY_SEED_KEYS[model.family], training.seed)
    return params.get(FAMILY_SEED_KEYS[model.family])


def _ensemble_hyperparameters(
    families: list[str], seed: int | None
) -> dict[str, dict[str, Any]]:
    """Per-member-family hyperparameters for an ENSEMBLE candidate.

    Per-family fixed hyperparameters inside an ensemble are out of scope: each
    member family uses AutoGluon's own defaults plus the shared seed.
    """
    return {
        family: ({FAMILY_SEED_KEYS[family]: seed} if seed is not None else {})
        for family in families
    }


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


def _check_trained_models(trained_models: list[str], family: str) -> None:
    if not trained_models:
        raise AutoGluonError("AutoGluon did not train a usable model")
    if any("WeightedEnsemble" in name for name in trained_models):
        raise AutoGluonError("AutoGluon unexpectedly trained a weighted ensemble")
    expected_tokens = FAMILY_MODEL_TOKENS[family]
    unrelated = [
        name for name in trained_models if not any(token in name for token in expected_tokens)
    ]
    if unrelated:
        raise AutoGluonError(
            "AutoGluon unexpectedly trained models outside family "
            f'{family}: {", ".join(unrelated)}'
        )


# Exact shapes observed from real AutoGluon 1.6 ensemble runs (Titanic and the
# regression/multiclass smoke tests): a stack level suffix ("_L2", "_L3", ...)
# on the weighted ensemble, an optional bag-fold group ("_BAG_L1", "_BAG_L2",
# ...) on a member family that is only absent when num_bag_folds=0, and an
# optional "_FULL" suffix once refit_full collapses everything onto one model.
_WEIGHTED_ENSEMBLE_PATTERN = re.compile(r"^WeightedEnsemble_L\d+(_FULL)?$")


def _member_family_pattern(token: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(token)}(_BAG_L\d+)?(_FULL)?$")


def _check_ensemble_trained_models(trained_models: list[str], families: list[str]) -> None:
    """The ENSEMBLE-candidate guard: only the configured member families (bagged,
    stacked, or refit ``_FULL`` variants) and AutoGluon's own weighted ensemble
    are allowed, matched by exact pattern rather than substring; a foreign
    family or an unrecognized name shape is still rejected.
    """
    if not trained_models:
        raise AutoGluonError("AutoGluon did not train a usable model")
    member_patterns = [
        _member_family_pattern(token)
        for family in families
        for token in FAMILY_MODEL_TOKENS[family]
    ]
    unrelated = [
        name
        for name in trained_models
        if not _WEIGHTED_ENSEMBLE_PATTERN.fullmatch(name)
        and not any(pattern.fullmatch(name) for pattern in member_patterns)
    ]
    if unrelated:
        raise AutoGluonError(
            "AutoGluon unexpectedly trained models outside the configured ensemble "
            f'families {sorted(families)}: {", ".join(unrelated)}'
        )


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
        is_ensemble = model.family == "ENSEMBLE"
        predictor_kwargs: dict[str, Any] = {
            "label": task.target,
            "problem_type": task.type,
            "eval_metric": AUTOGLUON_METRICS[primary_metric],
            "path": str(predictor_path),
            "verbosity": 2,
        }
        if task.type == "binary" and task.positive_class is not None:
            predictor_kwargs["positive_class"] = task.positive_class
        if training.seed is not None:
            predictor_kwargs["learner_kwargs"] = {"random_state": training.seed}

        if is_ensemble:
            hyperparameters = _ensemble_hyperparameters(model.params["families"], training.seed)
        else:
            model_params = dict(model.params)
            if training.seed is not None:
                # An explicitly fixed model seed in the config wins over training.seed.
                model_params.setdefault(FAMILY_SEED_KEYS[model.family], training.seed)
            assert model_params.get(FAMILY_SEED_KEYS[model.family]) == effective_model_seed(
                model, training
            )
            hyperparameters = {model.family: model_params}

        fit_kwargs: dict[str, Any] = {
            "train_data": train_data.copy(deep=True),
            "hyperparameters": hyperparameters,
            "hyperparameter_tune_kwargs": None,
            "fit_weighted_ensemble": is_ensemble,
            "fit_full_last_level_weighted_ensemble": False,
            "full_weighted_ensemble_additionally": False,
            "num_bag_folds": model.params["num_bag_folds"] if is_ensemble else 0,
            "num_stack_levels": model.params["num_stack_levels"] if is_ensemble else 0,
            "dynamic_stacking": False,
            "num_gpus": 0,
            "fit_strategy": "sequential",
        }
        if is_ensemble:
            # Memory safety: never let AutoGluon auto-pick parallel (Ray-based) fold
            # fitting, which can multiply peak RAM by the fold-parallelism factor.
            fit_kwargs["ag_args_ensemble"] = {"fold_fitting_strategy": "sequential_local"}
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
        fit_kwargs.update(resolve_resource_limits(training).fit_kwargs())

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
            if is_ensemble:
                # The ensemble's "best hyperparameters" is the config that produced
                # it, not a fitted model's own hyperparameters (there is no single
                # one): the same shape a single family stores in this field.
                best_model = str(predictor.model_best)
                best_hyperparameters = dict(model.params)
            elif hpo is not None:
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
        if is_ensemble:
            _check_ensemble_trained_models(trained_models, model.params["families"])
        else:
            _check_trained_models(trained_models, model.family)

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

    def fit_final(
        self,
        *,
        train_data: pd.DataFrame,
        test_features: pd.DataFrame,
        task: TaskConfig,
        model: ModelConfig,
        best_hyperparameters: dict[str, Any],
        effective_seed: int | None,
        primary_metric: str,
        predictor_path: Path,
        training: TrainingConfig,
    ) -> AutoGluonOutput:
        """Phase 6: one fixed-hyperparameter fit, then exactly one inference call.

        No HPO. ``refit_full`` retrains the single fitted model on every provided
        row (AutoGluon otherwise keeps an internal holdout out of a non-bagged fit)
        with the same hyperparameters, and makes it the predictor's best model.
        For an ENSEMBLE candidate, ``best_hyperparameters`` is the ensemble config
        (families/num_bag_folds/num_stack_levels) selected during tuning, and
        ``refit_full`` collapses the whole bagged/stacked/weighted ensemble into a
        single refit ``_FULL`` predictor trained on every provided row.
        """
        is_ensemble = model.family == "ENSEMBLE"
        predictor_kwargs: dict[str, Any] = {
            "label": task.target,
            "problem_type": task.type,
            "eval_metric": AUTOGLUON_METRICS[primary_metric],
            "path": str(predictor_path),
            "verbosity": 2,
        }
        if task.type == "binary" and task.positive_class is not None:
            predictor_kwargs["positive_class"] = task.positive_class
        if training.seed is not None:
            predictor_kwargs["learner_kwargs"] = {"random_state": training.seed}

        if is_ensemble:
            hyperparameters = _ensemble_hyperparameters(
                best_hyperparameters["families"], effective_seed
            )
            resolved_hyperparameters = hyperparameters
        else:
            params = dict(best_hyperparameters)
            if effective_seed is not None:
                params[FAMILY_SEED_KEYS[model.family]] = effective_seed
            hyperparameters = {model.family: params}
            resolved_hyperparameters = params

        fit_kwargs: dict[str, Any] = {
            "train_data": train_data.copy(deep=True),
            "hyperparameters": hyperparameters,
            "hyperparameter_tune_kwargs": None,
            "fit_weighted_ensemble": is_ensemble,
            "fit_full_last_level_weighted_ensemble": False,
            "full_weighted_ensemble_additionally": False,
            "num_bag_folds": best_hyperparameters["num_bag_folds"] if is_ensemble else 0,
            "num_stack_levels": best_hyperparameters["num_stack_levels"] if is_ensemble else 0,
            "dynamic_stacking": False,
            "num_gpus": 0,
            "fit_strategy": "sequential",
            "refit_full": True,
            "set_best_to_refit_full": True,
        }
        if is_ensemble:
            fit_kwargs["ag_args_ensemble"] = {"fold_fitting_strategy": "sequential_local"}
        if training.time_limit_seconds is not None:
            fit_kwargs["time_limit"] = training.time_limit_seconds
        fit_kwargs.update(resolve_resource_limits(training).fit_kwargs())

        try:
            predictor = self._factory()(**predictor_kwargs)
            predictor.fit(**fit_kwargs)
            trained_models = list(predictor.model_names())
            if is_ensemble:
                _check_ensemble_trained_models(trained_models, best_hyperparameters["families"])
            else:
                _check_trained_models(trained_models, model.family)
            frame = test_features.copy(deep=True)
            probabilities = None
            if task.type in {"binary", "multiclass"}:
                probabilities = predictor.predict_proba(frame, as_multiclass=True)
                predictions = predictor.predict_from_proba(probabilities)
            else:
                predictions = predictor.predict(frame)
            best_model = str(predictor.model_best)
        except (Exception, SystemExit) as exc:
            raise AutoGluonError(str(exc)) from exc

        if not isinstance(predictions, pd.Series):
            predictions = pd.Series(predictions, index=test_features.index)
        if probabilities is not None and not isinstance(probabilities, pd.DataFrame):
            probabilities = pd.DataFrame(probabilities, index=test_features.index)
        if len(predictions) != len(test_features) or (
            probabilities is not None and len(probabilities) != len(test_features)
        ):
            raise AutoGluonError("AutoGluon returned the wrong number of test predictions")
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
            best_hyperparameters=resolved_hyperparameters,
        )
