"""The only Phase-4 module that directly calls AutoGluon APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import version
import json
from pathlib import Path
import re
from typing import Any, Callable

import pandas as pd

from mltool.config import FAMILY_SEED_KEYS, HpoConfig, ModelConfig, TaskConfig, TrainingConfig
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
# SKLEARN candidates train exactly MLTool's wrapper: the model, its HPO trials,
# and the refit_full copy (mltool.autogluon_sklearn.AG_NAME is "MLToolSklearn").
SKLEARN_MODEL_NAME = re.compile(r"^MLToolSklearn(/T\d+)?(_FULL)?$")


# AutoGluon 1.6 has no seed argument on TabularPredictor.fit. Reproducibility is
# controlled by (a) the learner's random_state (internal holdout split), set via
# TabularPredictor(learner_kwargs=...), and (b) each model's own seed
# hyperparameter, whose name is family specific (AbstractModel.seed_name).
# FAMILY_SEED_KEYS lives in mltool.config, which needs it to validate search_space.

# Families for which AutoGluon defines no default hyperparameter search space:
# with hyperparameter_tune_kwargs they train a single default model.
# SKLEARN has no search space unless the user declares one, like RF/XT.
NO_SEARCH_SPACE_FAMILIES = frozenset({"RF", "XT", "SKLEARN"})

# MLTool's hpo.searcher -> the installed AutoGluon's local (Ray-free) searcher
# (autogluon/core/searcher/searcher_factory.py:6-13). "random" is kept verbatim:
# LocalSequentialScheduler.get_searcher_ rewrites it to "local_random"
# (autogluon/core/scheduler/seq_scheduler.py:128-130).
AUTOGLUON_SEARCHERS = {"random": "random", "grid": "local_grid"}

# LocalRandomSearcher(random_seed=0) when no seed is passed
# (autogluon/core/searcher/local_random_searcher.py:22).
AUTOGLUON_DEFAULT_SEARCHER_SEED = 0


def searcher_seed(hpo: HpoConfig, training: TrainingConfig) -> int | None:
    """The seed of the random searcher: ``training.seed`` itself, identical for
    every candidate, so one family on two FeatureSets tries the same configurations.

    Without a ``training.seed`` it is AutoGluon's own default (0), as before this
    setting existed. ``None`` for ``grid``: LocalGridSearcher draws nothing at
    random (local_grid_searcher.py) and would silently ignore a seed.
    """
    if hpo.searcher == "grid":
        return None
    return AUTOGLUON_DEFAULT_SEARCHER_SEED if training.seed is None else training.seed


def autogluon_spaces(search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """MLTool's normalized specs as ``autogluon.common.space`` objects.

    ``Real(lower, upper, default=None, log=False)`` and ``Int(lower, upper,
    default=None)`` default to ``lower`` when no default is given
    (autogluon/common/space.py:107-117, 147-152); ``Categorical(*data)`` always
    defaults to its first value (space.py:62-64), so a declared default is moved
    to the front.
    """
    from autogluon.common import space

    spaces: dict[str, Any] = {}
    for key, spec in search_space.items():
        if spec["type"] == "real":
            spaces[key] = space.Real(
                spec["low"], spec["high"], default=spec.get("default"), log=spec["log"]
            )
        elif spec["type"] == "int":
            spaces[key] = space.Int(spec["low"], spec["high"], default=spec.get("default"))
        else:
            values = list(spec["values"])
            if "default" in spec:
                wanted = json.dumps(spec["default"])
                first = next(i for i, value in enumerate(values) if json.dumps(value) == wanted)
                values.insert(0, values.pop(first))
            spaces[key] = space.Categorical(*values)
    return spaces


def _default_search_space(family: str, task_type: str) -> dict[str, Any]:
    """The installed AutoGluon's default search space for one family."""
    if family == "GBM":
        from autogluon.tabular.models.lgb.hyperparameters.searchspaces import (
            get_default_searchspace,
        )

        return get_default_searchspace(problem_type=task_type)
    if family == "CAT":
        from autogluon.tabular.models.catboost.hyperparameters.searchspaces import (
            get_default_searchspace,
        )

        return get_default_searchspace(task_type)
    if family == "XGB":
        from autogluon.tabular.models.xgboost.hyperparameters.searchspaces import (
            get_default_searchspace,
        )

        return get_default_searchspace(problem_type=task_type)
    # RFModel._get_default_searchspace returns {} (rf/rf_model.py:114-120) and
    # XTModel subclasses RFModel (xt/xt_model.py:8).
    return {}


def _space_record(value: Any) -> dict[str, Any]:
    from autogluon.common import space

    if isinstance(value, space.Real):
        return {"type": "real", "low": value.lower, "high": value.upper, "log": value.log,
                "default": value.default}
    if isinstance(value, space.Categorical):
        return {"type": "categorical", "values": list(value.data), "default": value.default}
    if isinstance(value, space.Int):
        return {"type": "int", "low": value.lower, "high": value.upper, "default": value.default}
    raise AutoGluonError(f"unsupported AutoGluon search space: {value!r}")


def effective_search_space(model: ModelConfig, task_type: str) -> dict[str, dict[str, Any]]:
    """Everything HPO will search for this candidate, and where each range came from.

    Mirrors ``AbstractModel._get_search_space`` (autogluon/core/models/abstract/
    abstract_model.py:610-620): every key the user put in the model's
    hyperparameters -- a fixed value or a Space -- is removed from the default
    search space, and the remaining default ranges are still searched. So a
    declared ``search_space`` is merged into AutoGluon's default, not a replacement.
    ``default`` is the value AutoGluon tries first (random searcher only).
    """
    if model.family == "ENSEMBLE":
        return {}
    from autogluon.common import space

    user_keys = set(model.params) | set(model.search_space) | {FAMILY_SEED_KEYS[model.family]}
    effective = {
        key: {**_space_record(value), "source": "autogluon_default"}
        for key, value in _default_search_space(model.family, task_type).items()
        if isinstance(value, space.Space) and key not in user_keys
    }
    for key, value in autogluon_spaces(model.search_space).items():
        effective[key] = {**_space_record(value), "source": "user"}
    return dict(sorted(effective.items()))


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
    if model.family == "SKLEARN":
        from mltool.custom_models import seed_decision

        return seed_decision(model, training)[0]
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
    # HPO only: the trials tied on the best validation score (empty = no tie),
    # and the one AutoGluon itself had picked before MLTool's tie-break.
    tied_trials: list[str] = field(default_factory=list)
    autogluon_best_trial: str | None = None


def _check_trained_models(trained_models: list[str], family: str) -> None:
    if not trained_models:
        raise AutoGluonError("AutoGluon did not train a usable model")
    if any("WeightedEnsemble" in name for name in trained_models):
        raise AutoGluonError("AutoGluon unexpectedly trained a weighted ensemble")
    if family == "SKLEARN":
        unrelated = [name for name in trained_models if not SKLEARN_MODEL_NAME.fullmatch(name)]
        if unrelated:
            raise AutoGluonError(
                f'AutoGluon unexpectedly trained models outside family SKLEARN: {", ".join(unrelated)}'
            )
        return
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


TRIAL_NAME = re.compile(r"/T(\d+)$")


def break_ties_by_trial_number(predictor: Any) -> tuple[list[str], str]:
    """Make the lowest-numbered of the trials tied on the best score the best model.

    AutoGluon picks ``max(..., key=(val_score, -predict_time))``
    (tabular/trainer/abstract_trainer.py, ``get_model_best``), so an exact tie is
    decided by a wall-clock timing and same-seed runs can disagree. Among trials
    with exactly the best ``val_score`` (higher is better in AutoGluon), the
    lowest trial number wins instead, via ``set_model_best(..., save_trainer=True)``
    (tabular/predictor/predictor.py:4224) so the saved predictor agrees. Returns
    the tied trial names (empty when there is no tie; nothing is changed then)
    and AutoGluon's own pick.
    """
    autogluon_pick = str(predictor.model_best)
    model_info = predictor.info()["model_info"]
    scores = {
        name: model_info[name].get("val_score")
        for name in predictor.model_names()
        if TRIAL_NAME.search(name) and model_info.get(name, {}).get("val_score") is not None
    }
    if not scores:
        return [], autogluon_pick
    best = max(scores.values())
    tied = sorted(
        (name for name, score in scores.items() if score == best),
        key=lambda name: int(TRIAL_NAME.search(name).group(1)),
    )
    if len(tied) < 2:
        return [], autogluon_pick
    predictor.set_model_best(tied[0], save_trainer=True)
    return tied, autogluon_pick


def sklearn_hyperparameters(
    model: ModelConfig, params: dict[str, Any], seed: Any | None
) -> dict[Any, dict[str, Any]]:
    """``{MLToolSklearnModel: ...}``: the estimator's params plus the wrapper's
    private keys (the entrypoint to build it from, the input mode, and the seed
    MLTool sets through ``set_params(random_state=...)`` when it sets one)."""
    from mltool.autogluon_sklearn import MLToolSklearnModel
    from mltool.custom_models import ENTRYPOINT_KEY, INPUT_KEY, RANDOM_STATE_KEY, entrypoint_spec

    wrapper = {**params, ENTRYPOINT_KEY: entrypoint_spec(model), INPUT_KEY: model.input}
    if seed is not None and "random_state" not in params:
        wrapper[RANDOM_STATE_KEY] = seed
    return {MLToolSklearnModel: wrapper}


def _best_hyperparameters(predictor: Any, model_name: str) -> dict[str, Any]:
    info = predictor.info()["model_info"][model_name]
    # the SKLEARN wrapper's own keys describe how it runs, not the estimator
    return {k: v for k, v in info.get("hyperparameters", {}).items() if not k.startswith("mltool_")}


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
        elif model.family == "SKLEARN":
            spaces = autogluon_spaces(model.search_space) if hpo is not None else {}
            hyperparameters = sklearn_hyperparameters(
                model, {**model.params, **spaces}, effective_model_seed(model, training)
            )
        else:
            model_params = dict(model.params)
            if training.seed is not None:
                # An explicitly fixed model seed in the config wins over training.seed.
                model_params.setdefault(FAMILY_SEED_KEYS[model.family], training.seed)
            assert model_params.get(FAMILY_SEED_KEYS[model.family]) == effective_model_seed(
                model, training
            )
            if hpo is not None:
                # Only a search passes Space objects: AutoGluon does not reject
                # one without HPO, it would hand the object to the model as a value.
                model_params.update(autogluon_spaces(model.search_space))
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
                "searcher": AUTOGLUON_SEARCHERS[hpo.searcher],
            }
            if hpo.searcher == "random" and training.seed is not None:
                # Forwarded unchanged to LocalRandomSearcher(random_seed=...):
                # scheduler_factory.py (scheduler_params.update) -> seq_scheduler.py
                # get_searcher_ -> searcher_factory. Without a training.seed nothing
                # is passed, exactly as before.
                fit_kwargs["hyperparameter_tune_kwargs"]["search_options"] = {
                    "random_seed": searcher_seed(hpo, training)
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
            tied_trials: list[str] = []
            autogluon_best_trial = None
            if hpo is not None and not is_ensemble:
                # Before any prediction, so MLTool's validation scores come from
                # the trial that best_hyperparameters describes.
                tied_trials, autogluon_best_trial = break_ties_by_trial_number(predictor)
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
            tied_trials=tied_trials,
            autogluon_best_trial=autogluon_best_trial,
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
        elif model.family == "SKLEARN":
            hyperparameters = sklearn_hyperparameters(model, dict(best_hyperparameters), effective_seed)
            resolved_hyperparameters = dict(best_hyperparameters)
            if effective_seed is not None and "random_state" not in resolved_hyperparameters:
                resolved_hyperparameters["random_state"] = effective_seed  # via set_params
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
