"""Phase-9 cross-validated candidate evaluation.

A single 134-row validation holdout could not separate FeatureSets: differences
on it were mostly noise. With ``evaluation.cv`` configured, a candidate is
instead scored on every fold of the development set (the train + validation rows
of the existing split) and ranked by the mean.

The critical property is that no fold leaks. For every fold the external
preprocessor and the FeatureSet's plugins are refit on that fold's training rows
alone and then applied to the held-out rows, reusing the very same helpers
``finalize`` uses for its refit. The materialized ``.mltool/features/``
artifacts are deliberately NOT used here: their plugins were fitted on the whole
train split, which overlaps every fold's held-out rows.

The test rows are never part of the development set, so ``test_data_used``
stays false for ``train`` and ``tune``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from statistics import fmean, stdev
import sys
import tempfile
import time
from typing import Any, Callable

import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from mltool.autogluon_adapter import AutoGluonAdapter
from mltool.config import FeaturePluginConfig, FeatureSetConfig, MLToolConfig, ModelConfig
from mltool.data import DataLoadError, load_dataset
from mltool.evaluation import evaluate_predictions
from mltool.feature_materialization import (
    FeatureMaterializationError,
    build_feature_set_from_frames,
    validate_prepared_manifest,
)
from mltool.feature_plugins import FeaturePluginError
from mltool.preprocessing import PreprocessingError, preprocess_frames
from mltool.splitting import SplitError, split_dataset
from mltool.training import TrainingError, convert_regression_targets

# Split names used only inside a fold; they never reach an artifact.
FOLD_TRAIN = "fold_train"
FOLD_HOLDOUT = "fold_holdout"
# Keeps each repeat's fold assignment distinct but deterministic from split.random_seed.
REPEAT_SEED_STRIDE = 9973


class CrossValidationError(TrainingError):
    """An expected problem setting up or running cross-validation.

    A subclass of TrainingError so a failing fold fails just that candidate,
    through the handler train/tune already use.
    """


@dataclass(frozen=True)
class FoldSpec:
    repeat: int
    fold: int
    seed: int
    train_labels: tuple[Any, ...]
    holdout_labels: tuple[Any, ...]

    @property
    def label(self) -> str:
        return f"repeat {self.repeat + 1} fold {self.fold + 1}"


@dataclass
class CvPlan:
    """The development rows and the fold assignment every candidate shares."""

    development: pd.DataFrame
    test_labels: tuple[Any, ...]
    folds: list[FoldSpec]
    fingerprint: str
    stratified: bool
    n_folds: int
    n_repeats: int

    @property
    def total_fits(self) -> int:
        return len(self.folds)

    def summary(self) -> dict[str, Any]:
        return {
            "folds": self.n_folds,
            "repeats": self.n_repeats,
            "total_fits_per_candidate": self.total_fits,
            "stratified": self.stratified,
            "development_rows": len(self.development),
            "fold_fingerprint": self.fingerprint,
            "seeds": sorted({fold.seed for fold in self.folds}),
        }


@dataclass
class CvScore:
    metrics: dict[str, float]
    metric_std: dict[str, float]
    fold_metrics: list[dict[str, Any]]
    positive_class: Any | None
    trained_models: list[str]
    autogluon_version: str
    target_conversion_applied: bool
    seconds: float

    def as_record(self, plan: CvPlan) -> dict[str, Any]:
        return {
            **plan.summary(),
            "metric_std": self.metric_std,
            "fold_metrics": self.fold_metrics,
        }


def repeat_seed(base_seed: int, repeat: int) -> int:
    """Deterministic per-repeat seed derived from ``split.random_seed``."""
    return (base_seed + REPEAT_SEED_STRIDE * repeat) % (2**31 - 1)


def _fold_fingerprint(folds: list[FoldSpec], stratified: bool) -> str:
    payload = {
        "stratified": stratified,
        "folds": [
            {
                "repeat": fold.repeat,
                "fold": fold.fold,
                "seed": fold.seed,
                # The held-out labels alone pin the assignment down.
                "holdout": sorted(int(label) if hasattr(label, "__index__") else str(label)
                                  for label in fold.holdout_labels),
            }
            for fold in folds
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def build_cv_plan(config: MLToolConfig) -> CvPlan:
    """Reproduce the development split and lay out the folds.

    The split is reproduced with the same function, config and seed ``finalize``
    uses, so CV scores and the final refit are built from exactly the same rows.
    Before that, the current split settings must be the ones ``prepare`` used:
    reproducing the split with a different seed would put prepared test rows
    into the development set, and the row counts alone cannot tell.
    """
    cv = config.evaluation.cv
    if cv is None:
        raise CrossValidationError("cross-validation is not configured")

    try:
        manifest = validate_prepared_manifest(config)
    except FeatureMaterializationError as exc:
        raise CrossValidationError(str(exc)) from exc
    recorded = manifest["split"]

    try:
        dataset = load_dataset(config.data)
        splits = split_dataset(
            dataset.frame,
            target=config.task.target,
            task_type=config.task.type,
            config=config.split,
        )
    except (DataLoadError, SplitError) as exc:
        raise CrossValidationError(f"could not reproduce the split for cross-validation: {exc}") from exc

    development = pd.concat([splits.train, splits.validation])
    expected = recorded["train_rows"] + recorded["validation_rows"]
    if len(development) != expected or len(splits.test) != recorded["test_rows"]:
        raise CrossValidationError(
            "reproduced split does not match the prepared manifest: "
            f"train+validation {len(development)} vs {expected}, "
            f'test {len(splits.test)} vs {recorded["test_rows"]}; run "mltool prepare" again'
        )
    test_labels = tuple(splits.test.index)
    if set(development.index) & set(test_labels):
        raise CrossValidationError("development rows overlap the test split")

    stratified = config.task.type in {"binary", "multiclass"}
    target = development[config.task.target]
    folds: list[FoldSpec] = []
    for repeat in range(cv.repeats):
        seed = repeat_seed(config.split.random_seed, repeat)
        splitter = (
            StratifiedKFold(n_splits=cv.folds, shuffle=True, random_state=seed)
            if stratified
            else KFold(n_splits=cv.folds, shuffle=True, random_state=seed)
        )
        try:
            assignment = list(splitter.split(development, target if stratified else None))
        except ValueError as exc:
            raise CrossValidationError(
                f"cannot build {cv.folds} {'stratified ' if stratified else ''}folds over "
                f"{len(development)} development rows: {exc}"
            ) from exc
        for index, (train_positions, holdout_positions) in enumerate(assignment):
            folds.append(
                FoldSpec(
                    repeat=repeat,
                    fold=index,
                    seed=seed,
                    train_labels=tuple(development.index[train_positions]),
                    holdout_labels=tuple(development.index[holdout_positions]),
                )
            )

    return CvPlan(
        development=development,
        test_labels=test_labels,
        folds=folds,
        fingerprint=_fold_fingerprint(folds, stratified),
        stratified=stratified,
        n_folds=cv.folds,
        n_repeats=cv.repeats,
    )


def progress(message: str) -> None:
    """Fold-level progress goes to stderr so the command's report stays clean."""
    print(message, file=sys.stderr, flush=True)


def cost_warning(candidates: int, plan: CvPlan, phase: str) -> str:
    return (
        f"MLTool {phase}: cross-validation will fit {candidates} candidate(s) x "
        f"{plan.n_folds} folds x {plan.n_repeats} repeat(s) = "
        f"{candidates * plan.total_fits} models, sequentially."
    )


def cv_score_candidate(
    config: MLToolConfig,
    plan: CvPlan,
    spec: FeatureSetConfig,
    model: ModelConfig,
    plugin_catalog: dict[str, FeaturePluginConfig],
    *,
    adapter_factory: Callable[[], AutoGluonAdapter] | None = None,
    label: str = "",
    report: Callable[[str], None] | None = None,
) -> CvScore:
    """Score one candidate across every fold, refitting all state per fold."""
    adapter_factory = adapter_factory or AutoGluonAdapter
    announce = report if report is not None else progress
    target = config.task.target
    started = time.perf_counter()
    fold_metrics: list[dict[str, Any]] = []
    positive_class: Any | None = None
    trained_models: list[str] = []
    autogluon_version = ""
    converted = False

    for index, fold in enumerate(plan.folds, start=1):
        fold_started = time.perf_counter()
        fold_train = plan.development.loc[list(fold.train_labels)]
        fold_holdout = plan.development.loc[list(fold.holdout_labels)]

        # Every fitted transform is rebuilt from this fold's training rows only.
        try:
            preprocessed = preprocess_frames(
                {FOLD_TRAIN: fold_train, FOLD_HOLDOUT: fold_holdout},
                fit_split=FOLD_TRAIN,
                target=target,
                config=config.preprocessing.external,
                config_path=config.config_path,
            )
            materialized = build_feature_set_from_frames(
                config, preprocessed.frames, FOLD_TRAIN, spec, plugin_catalog
            )
        except (PreprocessingError, FeaturePluginError, FeatureMaterializationError) as exc:
            raise CrossValidationError(
                f'fold {fold.label} of feature set "{spec.name}" could not be built: {exc}'
            ) from exc

        fit_frame = materialized.frames[FOLD_TRAIN].copy(deep=True)
        holdout_frame = materialized.frames[FOLD_HOLDOUT].copy(deep=True)
        if config.task.type == "regression":
            frames = {FOLD_TRAIN: fit_frame, FOLD_HOLDOUT: holdout_frame}
            converted = convert_regression_targets(frames, target)
            fit_frame, holdout_frame = frames[FOLD_TRAIN], frames[FOLD_HOLDOUT]
        holdout_target = holdout_frame[target].copy(deep=True)
        holdout_features = holdout_frame.drop(columns=[target])

        # Fold predictors are scratch: nothing downstream loads them.
        with tempfile.TemporaryDirectory(prefix="mltool-cv-") as scratch:
            output = adapter_factory().fit_predict(
                train_data=fit_frame,
                validation_features=holdout_features,
                task=config.task,
                model=model,
                primary_metric=config.evaluation.primary_metric,
                predictor_path=Path(scratch) / "predictor",
                training=config.training,
                hpo=None,  # cross-validation never searches hyperparameters
            )
            metrics = evaluate_predictions(
                task_type=config.task.type,
                evaluation=config.evaluation,
                y_true=holdout_target,
                predictions=output.predictions,
                probabilities=output.probabilities,
                positive_class=output.positive_class,
            )
        positive_class = output.positive_class
        trained_models = list(output.trained_models)
        autogluon_version = output.autogluon_version
        seconds = float(time.perf_counter() - fold_started)
        fold_metrics.append(
            {
                "repeat": fold.repeat,
                "fold": fold.fold,
                "seed": fold.seed,
                "rows": {"train": len(fit_frame), "holdout": len(holdout_frame)},
                "metrics": metrics,
                "seconds": seconds,
            }
        )
        primary = metrics[config.evaluation.primary_metric]
        announce(
            f"  [cv] {label or model.name} {fold.label} "
            f"({index}/{len(plan.folds)}) "
            f"{config.evaluation.primary_metric}={primary:.6f} ({seconds:.2f}s)"
        )

    names = list(fold_metrics[0]["metrics"])
    means = {name: fmean(entry["metrics"][name] for entry in fold_metrics) for name in names}
    stds = {
        name: (stdev([entry["metrics"][name] for entry in fold_metrics])
               if len(fold_metrics) > 1 else 0.0)
        for name in names
    }
    return CvScore(
        metrics=means,
        metric_std=stds,
        fold_metrics=fold_metrics,
        positive_class=positive_class,
        trained_models=trained_models,
        autogluon_version=autogluon_version,
        target_conversion_applied=converted,
        seconds=float(time.perf_counter() - started),
    )
