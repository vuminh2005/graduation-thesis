"""The single AutoGluon model class MLTool trains for every SKLEARN candidate.

Part of the adapter layer: only this module and ``autogluon_adapter`` import
AutoGluon. It follows the ``AbstractModel`` contract of the installed
AutoGluon 1.6.3 (autogluon/core/models/abstract/abstract_model.py):

- ``_fit`` must call ``self.preprocess(X, ...)`` and set ``self.model``
  (abstract_model.py:1607-1631); ``is_train=True`` marks the fit-time call.
- ``_predict_proba`` returns a 1-D prediction for regression, a 1-D
  positive-class probability for binary and an (n, K) matrix for multiclass,
  via ``_convert_proba_to_unified_form`` (abstract_model.py:1857-1893). Labels
  arrive encoded as 0..K-1, so probability columns are aligned to that order.
- ``can_refit_full`` is True, as for AutoGluon's own sklearn wrappers
  (tabular/models/lr/lr_model.py:348-350): the estimator never sees validation
  data, so refit_full retrains it on every row it is given instead of
  duplicating the model (core/models/abstract/_tags.py).

Persistence: AutoGluon saves models with plain ``pickle`` (common/savers/
save_pkl.py:27), which stores classes by reference. An estimator built from the
user's file would then need that file to load. ``__getstate__`` therefore
stores the fitted estimator as cloudpickle bytes; its user-defined classes and
functions are serialized by value (their module is not importable, see
``mltool.custom_models``), so the saved predictor loads with MLTool installed
and the user's file gone. This class itself is pickled by reference as
``mltool.autogluon_sklearn.MLToolSklearnModel``.

Time limits: AutoGluon passes ``time_limit`` to ``_fit``, but a running sklearn
``fit`` cannot be interrupted, so it is ignored here.
"""

from __future__ import annotations

from typing import Any

import cloudpickle
import numpy as np
import pandas as pd
from autogluon.common.features.types import R_CATEGORY, R_FLOAT, R_INT
from autogluon.core.constants import BINARY, REGRESSION
from autogluon.core.models import AbstractModel

from mltool.custom_models import (
    ENTRYPOINT_KEY,
    INPUT_KEY,
    RANDOM_STATE_KEY,
    AutoInput,
    align_probabilities,
    build_estimator,
    estimator_params,
    load_entrypoint,
)

AG_NAME = "MLToolSklearn"
_BY_VALUE = "_mltool_estimator_cloudpickle"


class MLToolSklearnModel(AbstractModel):
    ag_key = "MLTOOL_SKLEARN"
    ag_name = AG_NAME  # models are named MLToolSklearn, .../T<n>, ..._FULL
    _supported_problem_types = ["binary", "multiclass", "regression"]
    _default_auxiliary_params_extra = dict(valid_raw_types=[R_INT, R_FLOAT, R_CATEGORY])

    def _more_tags(self) -> dict[str, Any]:
        return {"can_refit_full": True}

    def _input_mode(self) -> str:
        return self.params.get(INPUT_KEY, "auto")

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs: Any) -> Any:
        X = super()._preprocess(X, **kwargs)
        if self._input_mode() == "raw":
            return X
        if is_train:
            # learned on exactly the rows this model is fit on (a CV fold,
            # the train split, or train+validation at refit)
            self._mltool_auto = AutoInput().fit(X)
        return self._mltool_auto.transform(X)

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> None:
        params = self._get_model_params()
        X = self.preprocess(X, is_train=True)
        estimator = build_estimator(load_entrypoint(params[ENTRYPOINT_KEY]), estimator_params(params))
        seed = params.get(RANDOM_STATE_KEY)
        if seed is not None:
            estimator.set_params(random_state=seed)
        estimator.fit(X, np.asarray(y))
        self.model = estimator

    def _predict_proba(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        X = self.preprocess(X, **kwargs)
        if self.problem_type == REGRESSION:
            return np.asarray(self.model.predict(X), dtype=np.float64).reshape(-1)
        n_classes = 2 if self.problem_type == BINARY else self.num_classes
        proba = align_probabilities(
            self.model.predict_proba(X), getattr(self.model, "classes_", None), n_classes
        )
        return self._convert_proba_to_unified_form(proba)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        if state.get("model") is not None:
            state[_BY_VALUE] = cloudpickle.dumps(state["model"])
            state["model"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        payload = state.pop(_BY_VALUE, None)
        self.__dict__.update(state)
        if payload is not None:
            self.model = cloudpickle.loads(payload)
