"""Command-line entry points for MLTool Phase 1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mltool.config import ConfigError, MLToolConfig, load_config
from mltool.data import DataLoadError, load_dataset
from mltool.report import DatasetSummary, ValidationReport
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
'''


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mltool", description="Validate a tabular MLTool project"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="create a minimal MLTool project")
    subparsers.add_parser("validate", help="validate the configured dataset")
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            return init_project()
        return validate_project()
    except Exception as exc:  # Keep unexpected failures concise for CLI users.
        print(f"MLTool failed unexpectedly: {exc}", file=sys.stderr)
        return 1
