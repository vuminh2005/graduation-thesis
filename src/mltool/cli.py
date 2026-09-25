"""Command-line entry points for every MLTool command."""

from __future__ import annotations

import argparse
import functools
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
from mltool.finalize import (
    FinalizeError,
    finalize_experiment,
    load_finalize_input,
    load_persisted_final,
)
from mltool.training import TrainingError, load_persisted_leaderboard, train_experiment
from mltool.tuning import (
    TuningError,
    build_tuning_selection,
    load_persisted_tuning,
    tune_experiment,
)
from mltool import tracking
from mltool.registry import RegistryBlocked, RegistryError, register_final
from mltool.reporting import render_best, render_logs, render_status
from mltool.runner import STEPS, execute, plan_decisions, render_decisions, render_summary
from mltool.scoring import ScoringError, render_scoring_error, score
from mltool.state import StateError, TrackedRun
from mltool.validation import validate_dataset


_CURRENT_RUN: list[TrackedRun] = []


def _detail(**values: object) -> None:
    """Attach details to the SQLite row of the command currently running."""
    if _CURRENT_RUN:
        _CURRENT_RUN[-1].details.update(values)


def _record_mlflow_runs(
    config: MLToolConfig, phase_dir: str, keys: list[str], run_ids: list[str]
) -> None:
    """Best-effort trace file mapping each artifact to the MLflow run that logged it.

    Written after the phase directory is committed; a failure is never fatal.
    """
    directory = config.config_path.parent / ".mltool" / phase_dir
    if not run_ids or len(run_ids) != len(keys) or not directory.is_dir():
        return
    payload = {
        "tracking_uri": tracking.tracking_uri(config.config_path.parent),
        "experiment": tracking.experiment_name(config),
        "runs": dict(zip(keys, run_ids)),
    }
    try:
        (directory / "mlflow.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"Warning: could not record MLflow run ids: {exc}", file=sys.stderr)


def _warn_tracking(warning: str | None) -> None:
    if warning:
        print(f"Warning: {warning}", file=sys.stderr)


def tracked(command: str):
    """Record one SQLite ``runs`` row per invocation, whatever its outcome."""

    def decorate(function):
        @functools.wraps(function)
        def wrapper(config_path: Path = Path("mltool.yaml"), *args, **kwargs) -> int:
            root = Path(config_path).parent
            run = TrackedRun(project_root=root, command=command)
            _CURRENT_RUN.append(run)
            try:
                code = function(config_path, *args, **kwargs)
            except BaseException as exc:
                run.details.setdefault("error", str(exc))
                run.finish("FAILED", 1)
                raise
            finally:
                _CURRENT_RUN.pop()
            if code == 0:
                status = "SUCCEEDED"
            elif run.details.get("blocked"):
                status = "BLOCKED"
            else:
                status = "FAILED"
            run.finish(status, code)
            return code

        return wrapper

    return decorate


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

  # An ENSEMBLE candidate lets AutoGluon bag/stack/weighted-ensemble several
  # families as one more model option; it is scored on the same validation
  # split and competes normally, it is not a special "finalize" mode. Left
  # commented out: it costs noticeably more time and memory than a single
  # family. Uncomment to enable, all params below are the defaults.
  # - name: ag_ensemble
  #   family: ENSEMBLE
  #   params:
  #     families: [GBM, CAT, XGB, RF, XT]
  #     num_bag_folds: 3
  #     num_stack_levels: 1

evaluation:
  primary_metric: roc_auc
  secondary_metrics:
    - f1
    - accuracy

training:
  time_limit_seconds: null

# What `tune` searches and how long it may take; `mltool run` needs it too.
hpo:
  top_n: 3                   # best training candidates to tune
  num_trials: 10             # HPO trials per candidate
  time_limit_seconds: 300    # budget per candidate
'''


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mltool",
        description="Prepare feature experiments and train isolated AutoGluon candidates",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="create a minimal MLTool project")
    run = subparsers.add_parser(
        "run", help="run every step from validate to register, skipping what is fresh"
    )
    run.add_argument("--dry-run", action="store_true",
                     help="print each step's decision and reason; run nothing")
    run.add_argument("--until", choices=STEPS, default=None, help="stop after this step")
    run.add_argument("--refinalize", action="store_true",
                     help="re-run a stale finalize (evaluates the test split again)")
    subparsers.add_parser("validate", help="validate the configured dataset")
    subparsers.add_parser("prepare", help="validate, split, and prepare the dataset")
    subparsers.add_parser("features", help="materialize configured feature sets")
    subparsers.add_parser("plan", help="show the FeatureSet x model candidate matrix")
    subparsers.add_parser("train", help="train and evaluate configured candidates")
    subparsers.add_parser("leaderboard", help="show the persisted global leaderboard")
    subparsers.add_parser("tune", help="run HPO on the top training candidates and select one")
    subparsers.add_parser("tuning-leaderboard", help="show the persisted tuning leaderboard")
    finalize = subparsers.add_parser(
        "finalize", help="refit the selected configuration on train+validation and test it"
    )
    finalize.add_argument(
        "--force",
        action="store_true",
        help="run again although a final model exists (evaluates the test split again)",
    )
    subparsers.add_parser("final-result", help="show the persisted final model result")
    register = subparsers.add_parser(
        "register", help="copy .mltool/final into a new registry version"
    )
    register.add_argument(
        "--force",
        action="store_true",
        help="register although the final artifacts are stale (records the warning)",
    )
    subparsers.add_parser("status", help="show artifact freshness and last runs per phase")
    logs = subparsers.add_parser("logs", help="show the recorded run history")
    logs.add_argument("--limit", type=int, default=20, help="number of runs to show")
    subparsers.add_parser("best", help="show the finalized and latest registered model")
    scoring = subparsers.add_parser(
        "score", help="predict raw rows with a registered model, using only its registry files"
    )
    scoring.add_argument("--input", required=True, type=Path, help="raw rows (.csv or .parquet)")
    scoring.add_argument("--output", required=True, type=Path, help="predictions (.csv or .parquet)")
    scoring.add_argument("--version", type=int, default=None, help="registry version (default: latest)")
    scoring.add_argument(
        "--registry", type=Path, default=Path(".mltool/registry"),
        help="registry directory (default: ./.mltool/registry)",
    )
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


@tracked("validate")
def validate_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        _detail(error=str(exc))
        print(_config_error_report(str(exc)).render())
        return 2

    if not config.validation.enabled:
        _detail(error="validation is disabled")
        print(_config_error_report('validation is disabled by "validation.enabled"').render())
        return 2

    try:
        dataset = load_dataset(config.data)
    except DataLoadError as exc:
        _detail(error=str(exc))
        print(_load_error_report(config, str(exc)).render())
        return 2

    report = validate_dataset(config, dataset)
    _detail(errors=len(report.errors), warnings=len(report.warnings))
    print(report.render())
    return 0 if report.is_valid else 2


@tracked("prepare")
def prepare_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        _detail(error=str(exc))
        print(render_preparation_error(str(exc)))
        return 2

    try:
        dataset = load_dataset(config.data)
    except DataLoadError as exc:
        _detail(error=str(exc))
        print(render_preparation_error(str(exc)))
        return 2

    # Preparation always validates, even if validation.enabled is false. An
    # invalid dataset must never reach splitting or external preprocessing.
    validation_report = validate_dataset(config, dataset)
    if not validation_report.is_valid:
        _detail(error="dataset validation failed", errors=len(validation_report.errors))
        print(validation_report.render())
        return 2

    try:
        result = prepare_dataset(
            config,
            dataset,
            warning_messages=validation_report.warnings,
        )
    except PreparationError as exc:
        _detail(error=str(exc))
        print(render_preparation_error(str(exc)))
        return 2

    _detail(
        train_rows=result.train_rows,
        validation_rows=result.validation_rows,
        test_rows=result.test_rows,
    )
    print(result.render())
    return 0


@tracked("features")
def features_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        _detail(error=str(exc))
        print(render_feature_error(str(exc)))
        return 2

    try:
        result = materialize_feature_sets(config)
    except FeatureMaterializationError as exc:
        _detail(error=str(exc))
        print(render_feature_error(str(exc)))
        return 2

    _detail(feature_sets=[feature_set.name for feature_set in result.feature_sets])
    print(result.render())
    return 0


@tracked("plan")
def plan_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
        plan = build_experiment_plan(config)
    except (ConfigError, ExperimentError) as exc:
        _detail(error=str(exc))
        print(render_experiment_error("MLTool training plan", str(exc)))
        return 2
    _detail(candidates=len(plan.candidates), primary_metric=config.evaluation.primary_metric)
    print(plan.render())
    return 0


@tracked("train")
def train_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
        plan = build_experiment_plan(config)
        result = train_experiment(plan)
    except (ConfigError, ExperimentError, TrainingError) as exc:
        _detail(error=str(exc))
        print(render_experiment_error("MLTool candidate training", str(exc)))
        return 2
    run_ids, warning = tracking.log_training(config, result)
    _warn_tracking(warning)
    _detail(
        candidates=len(getattr(result, "candidate_results", [])),
        succeeded=getattr(result, "successful_count", None),
        failed=getattr(result, "failed_count", None),
        primary_metric=config.evaluation.primary_metric,
        mlflow_runs=len(run_ids),
    )
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


@tracked("tune")
def tune_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
        plan = build_experiment_plan(config)
        selection = build_tuning_selection(plan)
        result = tune_experiment(plan, selection)
    except (ConfigError, ExperimentError, TrainingError, TuningError) as exc:
        _detail(error=str(exc))
        print(render_experiment_error("MLTool hyperparameter tuning", str(exc)))
        return 2
    run_ids, warning = tracking.log_tuning(config, result)
    _warn_tracking(warning)
    _record_mlflow_runs(
        config,
        "tuning",
        [r["candidate_id"] for r in getattr(result, "candidate_results", [])],
        run_ids,
    )
    selected = getattr(result, "selected", None)
    _detail(
        candidates=len(getattr(result, "candidate_results", [])),
        succeeded=getattr(result, "successful_count", None),
        failed=getattr(result, "failed_count", None),
        primary_metric=config.evaluation.primary_metric,
        selected_candidate_id=selected["candidate_id"] if selected else None,
        mlflow_runs=len(run_ids),
    )
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


@tracked("finalize")
def finalize_project(config_path: Path = Path("mltool.yaml"), force: bool = False) -> int:
    _detail(forced=force)
    existing = Path(config_path).parent / ".mltool" / "final" / "manifest.json"
    if existing.is_file() and not force:
        # Blocked before any work: the test split is evaluated once per final model.
        message = (
            "a final model already exists (.mltool/final/manifest.json). finalize evaluates "
            "the test split and should run once; run \"mltool register\" to keep the current "
            "result, then pass --force to refit and evaluate again"
        )
        _detail(blocked=True, error=message)
        print(render_experiment_error("MLTool final model", message))
        return 2
    try:
        config = load_config(config_path)
        plan = build_experiment_plan(config)
        finalize_input = load_finalize_input(plan)
        result = finalize_experiment(plan, finalize_input)
    except (ConfigError, ExperimentError, TrainingError, TuningError, FinalizeError) as exc:
        _detail(error=str(exc))
        print(render_experiment_error("MLTool final model", str(exc)))
        return 2
    run_ids, warning = tracking.log_final(config, result)
    _warn_tracking(warning)
    final = getattr(result, "result", {})
    _record_mlflow_runs(config, "final", [final.get("candidate_id", "final")], run_ids)
    _detail(
        candidate_id=final.get("candidate_id"),
        primary_metric=config.evaluation.primary_metric,
        test_metrics=final.get("metrics"),
        mlflow_runs=len(run_ids),
    )
    print(result.render())
    return 0


def final_result_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
        final = load_persisted_final(config)
    except (ConfigError, FinalizeError) as exc:
        print(render_experiment_error("MLTool final result", str(exc)))
        return 2
    print(final.render())
    return 0


@tracked("register")
def register_project(config_path: Path = Path("mltool.yaml"), force: bool = False) -> int:
    _detail(forced=force)
    try:
        config = load_config(config_path)
        registered = register_final(config, force=force)
    except RegistryBlocked as exc:
        _detail(blocked=True, error=str(exc))
        print(render_experiment_error("MLTool model registry", str(exc)))
        return 2
    except (ConfigError, RegistryError) as exc:
        _detail(error=str(exc))
        print(render_experiment_error("MLTool model registry", str(exc)))
        return 2
    _detail(
        version=registered.version,
        candidate_id=registered.metadata["selected"]["candidate_id"],
        stale_warning=registered.metadata["warning"],
    )
    print(registered.render())
    return 0


def status_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
        print(render_status(config))
    except (ConfigError, StateError) as exc:
        print(render_experiment_error("MLTool status", str(exc)))
        return 2
    return 0


def logs_project(config_path: Path = Path("mltool.yaml"), limit: int | None = 20) -> int:
    try:
        load_config(config_path)
        print(render_logs(Path(config_path).parent, limit))
    except (ConfigError, StateError) as exc:
        print(render_experiment_error("MLTool run history", str(exc)))
        return 2
    return 0


def best_project(config_path: Path = Path("mltool.yaml")) -> int:
    try:
        config = load_config(config_path)
        print(render_best(config))
    except (ConfigError, FinalizeError) as exc:
        print(render_experiment_error("MLTool best configuration", str(exc)))
        return 2
    return 0


def score_project(
    *, input_path: Path, output_path: Path, version: int | None = None,
    registry: Path = Path(".mltool/registry"),
) -> int:
    """Not tracked: it writes only the requested output file, never project state."""
    try:
        result = score(
            registry=registry, input_path=input_path, output_path=output_path, version=version
        )
    except ScoringError as exc:
        print(render_scoring_error(str(exc)))
        return 2
    print(result.render())
    return 0


# Looked up at call time, so each step is exactly the command of the same name.
RUN_STEPS = {
    "validate": lambda path, **kw: validate_project(path, **kw),
    "prepare": lambda path, **kw: prepare_project(path, **kw),
    "features": lambda path, **kw: features_project(path, **kw),
    "plan": lambda path, **kw: plan_project(path, **kw),
    "train": lambda path, **kw: train_project(path, **kw),
    "tune": lambda path, **kw: tune_project(path, **kw),
    "finalize": lambda path, **kw: finalize_project(path, **kw),
    "register": lambda path, **kw: register_project(path, **kw),  # never forced
}


def run_project(
    config_path: Path = Path("mltool.yaml"), *, dry_run: bool = False,
    until: str | None = None, refinalize: bool = False,
) -> int:
    if dry_run:  # writes nothing, so like status it is not recorded
        try:
            config = load_config(config_path)
            print(render_decisions(plan_decisions(config, until=until, refinalize=refinalize)))
        except (ConfigError, StateError) as exc:
            print(render_experiment_error("MLTool run", str(exc)))
            return 2
        return 0
    return _run_pipeline(config_path, until=until, refinalize=refinalize)


@tracked("run")
def _run_pipeline(config_path: Path, *, until: str | None, refinalize: bool) -> int:
    _detail(until=until, refinalize=refinalize)
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        _detail(error=str(exc))
        print(render_experiment_error("MLTool run", str(exc)))
        return 2
    report = execute(config_path, config, RUN_STEPS, until=until, refinalize=refinalize)
    _detail(**report.details())
    if report.stopped_at:
        _detail(blocked=True)
    elif report.failed_step:
        _detail(error=f"step {report.failed_step} failed with exit code {report.exit_code}")
    final = None
    if (config.config_path.parent / ".mltool/final/manifest.json").is_file():
        try:
            final = load_persisted_final(config)
        except FinalizeError:
            final = None
    print(render_summary(config, report, final))
    return report.exit_code


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            return init_project()
        if args.command == "run":
            return run_project(dry_run=args.dry_run, until=args.until, refinalize=args.refinalize)
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
        if args.command == "tuning-leaderboard":
            return tuning_leaderboard_project()
        if args.command == "finalize":
            return finalize_project(force=args.force)
        if args.command == "final-result":
            return final_result_project()
        if args.command == "register":
            return register_project(force=args.force)
        if args.command == "status":
            return status_project()
        if args.command == "logs":
            return logs_project(limit=args.limit)
        if args.command == "score":
            return score_project(
                input_path=args.input, output_path=args.output, version=args.version,
                registry=args.registry,
            )
        return best_project()
    except Exception as exc:  # Keep unexpected failures concise for CLI users.
        print(f"MLTool failed unexpectedly: {exc}", file=sys.stderr)
        return 1
