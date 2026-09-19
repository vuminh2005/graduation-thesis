"""Command-line entry points for MLTool Phases 1 through 5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mltool.config import ConfigError, MLToolConfig, load_config
from mltool.data import DataLoadError, load_dataset
from mltool.feature_materialization import (
    FeatureMaterializationError,
    materialize_feature_sets,
    render_feature_error,
)
from mltool.experiment import (
    ExperimentError,
    build_experiment_plan,
    render_experiment_error,
)
from mltool.preparation import PreparationError, prepare_dataset, render_preparation_error
from mltool.report import DatasetSummary, ValidationReport
from mltool.training import TrainingError, load_persisted_leaderboard, train_experiment
from mltool.tuning import (
    TuningError,
    build_tuning_selection,
    load_persisted_tuning,
    tune_experiment,
)
from mltool.validation import validate_dataset


CONFIG_TEMPLATE = '''schema_version: "0.1"

project:
  name: {project_name}

task:
  type: binary
  target: label
  positive_class: 1

data:
  format: auto
  path: ./data/dataset.csv

validation:
  enabled: true
  fail_on_error: true

split:
  validation_ratio: 0.15
  test_ratio: 0.15
  stratify: auto
  random_seed: 42

preprocessing:
  external:
    enabled: false
    entrypoint: null
    params: {{}}

features:
  plugins: []

  sets:
    - name: base
      source_columns:
        - "*"
      plugins: []

models:
  - name: lightgbm
    family: GBM
    params: {{}}

  - name: random_forest
    family: RF
    params: {{}}

evaluation:
  primary_metric: roc_auc
  secondary_metrics:
    - f1
    - accuracy

training:
  time_limit_seconds: null
'''


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mltool",
        description="Prepare feature experiments and train isolated AutoGluon candidates",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="create a minimal MLTool project")
    subparsers.add_parser("validate", help="validate the configured dataset")
    subparsers.add_parser("prepare", help="validate, split, and prepare the dataset")
    subparsers.add_parser("features", help="materialize configured feature sets")
    subparsers.add_parser("plan", help="show the FeatureSet x model candidate matrix")
    subparsers.add_parser("train", help="train and evaluate configured candidates")
    subparsers.add_parser("leaderboard", help="show the persisted global leaderboard")
    subparsers.add_parser("tune", help="run HPO on the top training candidates and select one")
    subparsers.add_parser("tuning-leaderboard", help="show the persisted tuning leaderboard")
    return parser


def init_project(directory: Path | None = None) -> int:
    root = (directory or Path.cwd()).resolve()
    config_path = root / "mltool.yaml"
    if config_path.exists():
        print(f"Error: {config_path} already exists; refusing to overwrite it", file=sys.stderr)
        return 2

    try:
        (root / "data").mkdir(exist_ok=True)
        project_name = root.name or "mltool-project"
        config_path.write_text(
            CONFIG_TEMPLATE.format(project_name=json.dumps(project_name)), encoding="utf-8"
        )
    except OSError as exc:
        print(f"Error: could not initialize MLTool project: {exc}", file=sys.stderr)
        return 1

    print(f"Created {config_path}")
    print(f"Created {root / 'data'}")
    return 0


def _config_error_report(message: str) -> ValidationReport:
    report = ValidationReport()
    report.errors.append(message)
    return report


def _load_error_report(config: MLToolConfig, message: str) -> ValidationReport:
    report = ValidationReport(
        project_name=config.project.name,
        task_type=config.task.type,
        target=config.task.target,
        dataset=DatasetSummary(path=config.data.path),
    )
    report.errors.append(message)
    return report


def validate_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(_config_error_report(str(exc)).render())
        return 2

    if not config.validation.enabled:
        print(_config_error_report('validation is disabled by "validation.enabled"').render())
        return 2

    try:
        dataset = load_dataset(config.data)
    except DataLoadError as exc:
        print(_load_error_report(config, str(exc)).render())
        return 2

    report = validate_dataset(config, dataset)
    print(report.render())
    return 0 if report.is_valid else 2


def prepare_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(render_preparation_error(str(exc)))
        return 2

    try:
        dataset = load_dataset(config.data)
    except DataLoadError as exc:
        print(render_preparation_error(str(exc)))
        return 2

    # Preparation always validates, even if validation.enabled is false. An
    # invalid dataset must never reach splitting or external preprocessing.
    validation_report = validate_dataset(config, dataset)
    if not validation_report.is_valid:
        print(validation_report.render())
        return 2

    try:
        result = prepare_dataset(
            config,
            dataset,
            warning_messages=validation_report.warnings,
        )
    except PreparationError as exc:
        print(render_preparation_error(str(exc)))
        return 2

    print(result.render())
    return 0


def features_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(render_feature_error(str(exc)))
        return 2

    try:
        result = materialize_feature_sets(config)
    except FeatureMaterializationError as exc:
        print(render_feature_error(str(exc)))
        return 2

    print(result.render())
    return 0


def plan_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
        plan = build_experiment_plan(config)
    except (ConfigError, ExperimentError) as exc:
        print(render_experiment_error("MLTool training plan", str(exc)))
        return 2
    print(plan.render())
    return 0


def train_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
        plan = build_experiment_plan(config)
        result = train_experiment(plan)
    except (ConfigError, ExperimentError, TrainingError) as exc:
        print(render_experiment_error("MLTool candidate training", str(exc)))
        return 2
    print(result.render())
    return 0 if result.is_successful else 1


def leaderboard_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
        leaderboard = load_persisted_leaderboard(config)
    except (ConfigError, TrainingError) as exc:
        print(render_experiment_error("MLTool global leaderboard", str(exc)))
        return 2
    print(leaderboard.render())
    return 0


def tune_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
        plan = build_experiment_plan(config)
        selection = build_tuning_selection(plan)
        result = tune_experiment(plan, selection)
    except (ConfigError, ExperimentError, TrainingError, TuningError) as exc:
        print(render_experiment_error("MLTool hyperparameter tuning", str(exc)))
        return 2
    print(result.render())
    return 0 if result.is_successful else 1


def tuning_leaderboard_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
        tuning = load_persisted_tuning(config)
    except (ConfigError, TuningError) as exc:
        print(render_experiment_error("MLTool tuning leaderboard", str(exc)))
        return 2
    print(tuning.render())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            return init_project()
        if args.command == "validate":
            return validate_project()
        if args.command == "prepare":
            return prepare_project()
        if args.command == "features":
            return features_project()
        if args.command == "plan":
            return plan_project()
        if args.command == "train":
            return train_project()
        if args.command == "leaderboard":
            return leaderboard_project()
        if args.command == "tune":
            return tune_project()
        return tuning_leaderboard_project()
    except Exception as exc:  # Keep unexpected failures concise for CLI users.
        print(f"MLTool failed unexpectedly: {exc}", file=sys.stderr)
        return 1
