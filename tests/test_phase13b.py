"""Phase 13b: the zero-effort path works for every task type.

After ``mltool init`` a user sets only task.type and task.target (and puts the
data at data.path). Default secondary metrics never collide with a chosen
primary. Declared identifier columns are not features, and ``validate`` warns
about columns that look like identifiers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from mltool.cli import features_project, init_project, plan_project, prepare_project, run_project
from mltool.config import ConfigError, load_config
from mltool.data import load_dataset
from mltool.experiment import ExperimentError, build_experiment_plan
from mltool.registry import list_versions
from mltool.validation import validate_dataset
from test_phase12 import fake_adapters  # noqa: F401  (autouse fixture: fake fits)
from test_phase4 import project_config, write_project

TASKS = ["binary", "multiclass", "regression"]


def frame(task: str, rows: int = 120) -> pd.DataFrame:
    x = [(i * 37) % 101 for i in range(rows)]
    label = {
        "binary": [int(v > 50) for v in x],
        "multiclass": [min(2, v // 34) for v in x],
        "regression": [float(v) * 1.5 + (i % 7) for i, v in enumerate(x)],
    }[task]
    return pd.DataFrame({"x": x, "z": [(i * 11) % 17 for i in range(rows)], "outcome": label})


def zero_effort(root: Path, task: str, *, with_task_flag: bool) -> Path:
    """init, the data at the template's path, then only type and target edited (as sed would)."""
    assert init_project(root, **({"task": task} if with_task_flag else {})) == 0
    frame(task).to_csv(root / "data/dataset.csv", index=False)
    path = root / "mltool.yaml"
    template = path.read_text().splitlines()
    edited = [
        f"  type: {task}" if line.startswith("  type: ") else
        "  target: outcome" if line.startswith("  target: ") else line
        for line in template
    ]
    path.write_text("\n".join(edited) + "\n")
    changed = [a for a, b in zip(template, edited) if a != b]
    assert [line.split(":")[0] for line in changed] == ["  type", "  target"]  # nothing else
    return path


@pytest.mark.parametrize("with_task_flag", [False, True], ids=["plain-init", "init--task"])
@pytest.mark.parametrize("task", TASKS)
def test_setting_only_type_and_target_is_enough(tmp_path: Path, task: str, with_task_flag: bool) -> None:
    path = zero_effort(tmp_path, task, with_task_flag=with_task_flag)
    config = load_config(path)
    assert config.task.type == task and config.task.positive_class is None
    assert config.training.seed == 42
    assert prepare_project(path) == 0 and features_project(path) == 0 and plan_project(path) == 0
    assert run_project(path) == 0  # and on to a registered model (fake fits)
    assert list_versions(tmp_path) == [1]


def test_init_task_prefills_the_type_and_keeps_binary_as_default(tmp_path: Path) -> None:
    (tmp_path / "plain").mkdir()
    assert init_project(tmp_path / "plain") == 0
    assert "  type: binary  #" in (tmp_path / "plain/mltool.yaml").read_text()
    (tmp_path / "reg").mkdir()
    assert init_project(tmp_path / "reg", task="regression") == 0
    text = (tmp_path / "reg/mltool.yaml").read_text()
    assert "  type: regression  #" in text
    for commented in ("# positive_class: 1", "# id_columns: [id]", "# evaluation:", "#   primary_metric:"):
        assert commented in text, commented


# =========================== default secondary metrics ===========================


def metrics(tmp_path: Path, task: str, evaluation: dict[str, Any] | None) -> tuple[str, list[str]]:
    raw = project_config(task=task, positive_class=None, evaluation=evaluation)
    config = load_config(write_project(tmp_path, raw=raw))
    return config.evaluation.primary_metric, config.evaluation.secondary_metrics


@pytest.mark.parametrize(
    ("task", "evaluation", "expected"),
    [
        ("regression", None, ("rmse", ["mae", "r2"])),                       # unchanged
        ("regression", {"primary_metric": "r2"}, ("r2", ["mae"])),           # was an error
        ("regression", {"primary_metric": "mae"}, ("mae", ["r2"])),          # was an error
        ("binary", {"primary_metric": "accuracy"}, ("accuracy", ["f1"])),    # was an error
        ("binary", {"primary_metric": "log_loss"}, ("log_loss", ["f1", "accuracy"])),  # unchanged
        ("multiclass", {"primary_metric": "f1_macro"}, ("f1_macro", [])),    # was an error
    ],
)
def test_default_secondaries_exclude_the_chosen_primary(
    tmp_path: Path, task: str, evaluation: dict[str, Any] | None, expected: tuple[str, list[str]]
) -> None:
    assert metrics(tmp_path, task, evaluation) == expected


def test_an_explicit_duplicate_is_still_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must not repeat the primary metric"):
        metrics(tmp_path, "regression", {"primary_metric": "r2", "secondary_metrics": ["mae", "r2"]})


# =========================== identifier columns ==================================


def id_project(root: Path, id_columns: list[str] | None) -> Path:
    data = pd.DataFrame({"row_id": range(1000, 1100), "x": range(100),
                         "z": [i % 7 for i in range(100)], "label": [i % 2 for i in range(100)]})
    raw = project_config(models=[{"name": "gbm", "family": "GBM", "params": {}}])
    if id_columns is not None:
        raw["data"]["id_columns"] = id_columns
    path = write_project(root, frame=data, raw=raw)
    assert prepare_project(path) == 0 and features_project(path) == 0
    return path


def test_star_source_columns_exclude_declared_id_columns(tmp_path: Path) -> None:
    with_id = id_project(tmp_path / "declared", ["row_id"])
    plan = build_experiment_plan(load_config(with_id))
    assert plan.feature_sets[0].feature_columns == ["x", "z"]
    without = id_project(tmp_path / "undeclared", None)
    assert build_experiment_plan(load_config(without)).feature_sets[0].feature_columns == ["row_id", "x", "z"]


def test_changing_id_columns_makes_star_feature_sets_stale(tmp_path: Path) -> None:
    """No special case: the resolved recipe's source columns change, so the
    existing recipe rule reports the features stale."""
    path = id_project(tmp_path, None)
    raw = yaml.safe_load(path.read_text())
    raw["data"]["id_columns"] = ["row_id"]
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    with pytest.raises(ExperimentError, match='feature set "base" source columns changed'):
        build_experiment_plan(load_config(path))
    assert features_project(path) == 0
    assert build_experiment_plan(load_config(path)).feature_sets[0].feature_columns == ["x", "z"]


def warnings_for(tmp_path: Path, data: pd.DataFrame, id_columns: list[str] | None = None) -> list[str]:
    raw = project_config()
    if id_columns:
        raw["data"]["id_columns"] = id_columns
    config = load_config(write_project(tmp_path, frame=data, raw=raw))
    return [w for w in validate_dataset(config, load_dataset(config.data)).warnings if "identif" in w]


def test_the_identifier_warning_fires_on_id_like_columns_only(tmp_path: Path) -> None:
    n = 60
    data = pd.DataFrame({
        "id": [i % 5 for i in range(n)],                  # named like one (values repeat)
        "Customer_ID": [i % 3 for i in range(n)],         # *_id, any case
        "ticket": [f"T{i:04d}" for i in range(n)],        # distinct text
        "serial": [7000 + i for i in range(n)],           # distinct integers
        "amount": [i * 1.37 for i in range(n)],           # distinct floats: a normal feature
        "count": [i % 9 for i in range(n)],               # repeating integers
        "idle": [i % 4 for i in range(n)],                # "id" prefix is not an id name
        "label": [i % 2 for i in range(n)],
    })
    flagged = sorted(w.split('"')[1] for w in warnings_for(tmp_path / "all", data))
    assert flagged == ["Customer_ID", "id", "serial", "ticket"]
    # declared identifiers are not warned about, and it is never an error
    config_warnings = warnings_for(tmp_path / "declared", data, ["id", "serial"])
    assert sorted(w.split('"')[1] for w in config_warnings) == ["Customer_ID", "ticket"]


def test_the_warning_suggests_id_columns_and_does_not_block(tmp_path: Path) -> None:
    data = pd.DataFrame({"id": range(40), "x": [i % 5 for i in range(40)], "label": [i % 2 for i in range(40)]})
    raw = project_config()
    config = load_config(write_project(tmp_path, frame=data, raw=raw))
    report = validate_dataset(config, load_dataset(config.data))
    assert report.is_valid
    assert report.warnings == [
        'column "id" is named like an identifier; if it identifies rows rather than describing '
        "them, declare it in data.id_columns so it is not used as a feature"
    ]


def test_the_cli_accepts_init_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mltool.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["init", "--task", "multiclass"]) == 0
    assert load_config(tmp_path / "mltool.yaml").task.type == "multiclass"
