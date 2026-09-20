"""Regressions for the four defects found in the whole-system report of f0ab0e7."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from mltool.cli import features_project, plan_project, prepare_project, validate_project
from mltool.config import ConfigError, load_config
from test_phase4 import project_config, write_project

MODELS = [{"name": "gbm", "family": "GBM", "params": {}}]


def loaded(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "mltool.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


# --- Fix 1: feature-less dataset ---------------------------------------------------


def test_target_only_dataset_is_caught_by_validate_features_and_never_reaches_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = pd.DataFrame({"label": [i % 2 for i in range(100)]})
    path = write_project(tmp_path, frame=frame, raw=project_config(models=MODELS))

    assert validate_project(path) == 0  # still valid, but no longer silent
    out = capsys.readouterr().out
    assert "no columns other than the target" in out

    assert prepare_project(path) == 0
    capsys.readouterr()
    assert features_project(path) == 2
    out = capsys.readouterr().out
    assert 'FeatureSet "base" has no source columns' in out
    assert "no columns other than the target" in out
    assert "run \"mltool features\" again" not in out  # the old, wrong advice
    assert not (tmp_path / ".mltool/features").exists()  # no half-built artifact

    assert plan_project(path) == 2
    plan_out = capsys.readouterr().out
    assert "features" in plan_out and "run \"mltool features\" first" in plan_out
    assert "final schema metadata is invalid" not in plan_out


def test_dataset_with_features_gets_no_such_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_project(tmp_path, raw=project_config(models=MODELS))
    assert validate_project(path) == 0
    assert "no columns other than the target" not in capsys.readouterr().out


def test_explicit_source_columns_are_unaffected(tmp_path: Path) -> None:
    features = {"plugins": [], "sets": [{"name": "s", "source_columns": ["x"], "plugins": []}]}
    path = write_project(tmp_path, raw=project_config(models=MODELS, features=features))
    assert prepare_project(path) == 0
    assert features_project(path) == 0


# --- Fix 2: unknown keys ---------------------------------------------------------------


def with_change(mutate) -> dict[str, Any]:
    raw = project_config(
        models=copy.deepcopy(MODELS), evaluation={"primary_metric": "roc_auc"},
        training={"seed": 1},
    )
    raw["hpo"] = {"time_limit_seconds": 5}
    raw["features"] = {
        "plugins": [{"name": "p", "entrypoint": "./p.py:P", "params": {}}],
        "sets": [{"name": "s", "source_columns": ["*"], "plugins": ["p"]}],
    }
    mutate(raw)
    return raw


CASES = {
    "top-level": (lambda r: r.__setitem__("trainig", {"seed": 1}), "top-level", "trainig"),
    "project": (lambda r: r["project"].__setitem__("nam", "x"), "project", "nam"),
    "task": (lambda r: r["task"].__setitem__("targt", "oops"), "task", "targt"),
    "data": (lambda r: r["data"].__setitem__("paht", "x"), "data", "paht"),
    "validation": (lambda r: r["validation"].__setitem__("enable", True), "validation", "enable"),
    "split": (lambda r: r["split"].__setitem__("random_sed", 1), "split", "random_sed"),
    "preprocessing": (
        lambda r: r["preprocessing"].__setitem__("extrnal", {}), "preprocessing", "extrnal"
    ),
    "preprocessing.external": (
        lambda r: r["preprocessing"]["external"].__setitem__("enable", True),
        "preprocessing.external", "enable",
    ),
    "features": (lambda r: r["features"].__setitem__("set", []), "features", "set"),
    "features.plugins[0]": (
        lambda r: r["features"]["plugins"][0].__setitem__("param", {}),
        "features.plugins[0]", "param",
    ),
    "features.sets[0]": (
        lambda r: r["features"]["sets"][0].__setitem__("source_column", ["*"]),
        "features.sets[0]", "source_column",
    ),
    "models[0]": (lambda r: r["models"][0].__setitem__("famly", "GBM"), "models[0]", "famly"),
    "evaluation": (
        lambda r: r["evaluation"].__setitem__("primary_metrics", "x"), "evaluation",
        "primary_metrics",
    ),
    "training": (lambda r: r["training"].__setitem__("sed", 3), "training", "sed"),
    "hpo": (lambda r: r["hpo"].__setitem__("trials", 5), "hpo", "trials"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_misspelled_key_is_rejected_with_the_hpo_convention(tmp_path: Path, case: str) -> None:
    mutate, section, key = CASES[case]
    path = loaded(tmp_path, with_change(mutate))
    with pytest.raises(ConfigError) as info:
        load_config(path)
    message = str(info.value)
    assert f'unsupported "{section}" setting(s): {key}' == message, message


def test_several_unknown_keys_are_all_listed_sorted(tmp_path: Path) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["split"]["zzz"] = 1
        raw["split"]["aaa"] = 2

    with pytest.raises(ConfigError, match=r'unsupported "split" setting\(s\): aaa, zzz'):
        load_config(loaded(tmp_path, with_change(mutate)))


def test_typo_now_fails_at_the_cli_instead_of_being_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = with_change(lambda r: r.__setitem__("trainig", {"seed": 1}))
    path = loaded(tmp_path, raw)
    assert validate_project(path) == 2
    assert 'unsupported "top-level" setting(s): trainig' in capsys.readouterr().out


def test_every_documented_key_still_loads(tmp_path: Path) -> None:
    """Guards against the allow-lists being narrower than the real schema."""
    raw = with_change(lambda r: None)
    raw["validation"] = {"enabled": True, "fail_on_error": True}
    raw["split"] = {"validation_ratio": 0.2, "test_ratio": 0.2, "stratify": "auto",
                    "random_seed": 3}
    raw["preprocessing"] = {"external": {"enabled": True, "entrypoint": "./e.py:E", "params": {}}}
    raw["task"]["positive_class"] = 1
    raw["evaluation"] = {"primary_metric": "roc_auc", "secondary_metrics": ["f1"]}
    raw["training"] = {"time_limit_seconds": 5, "seed": 2}
    raw["hpo"] = {"top_n": 2, "num_trials": 3, "time_limit_seconds": 5}
    config = load_config(loaded(tmp_path, raw))
    assert config.hpo is not None and config.training.seed == 2


# --- Fix 3: plugin names ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["my/plugin", "has space", "../escape", "a.b", "-lead", ""])
def test_unsafe_plugin_names_fail_at_config_load(tmp_path: Path, name: str) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["features"]["plugins"][0]["name"] = name
        raw["features"]["sets"][0]["plugins"] = [name]

    with pytest.raises(ConfigError, match="(unsafe|required and must be a non-empty)"):
        load_config(loaded(tmp_path, with_change(mutate)))


def test_unsafe_plugin_name_is_reported_by_validate_not_by_finalize(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["features"]["plugins"][0]["name"] = "my/sq"
        raw["features"]["sets"][0]["plugins"] = ["my/sq"]

    path = loaded(tmp_path, with_change(mutate))
    assert validate_project(path) == 2
    out = capsys.readouterr().out
    assert 'feature plugin name "my/sq" is unsafe' in out


@pytest.mark.parametrize("name", ["sq", "Sq_2", "cross-term", "a1"])
def test_safe_plugin_names_still_load(tmp_path: Path, name: str) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        raw["features"]["plugins"][0]["name"] = name
        raw["features"]["sets"][0]["plugins"] = [name]

    assert load_config(loaded(tmp_path, with_change(mutate))).features.plugins[0].name == name


def test_plugin_names_use_exactly_the_set_and_model_rule() -> None:
    import mltool.config as config_module

    assert config_module.SAFE_PLUGIN_NAME is config_module.SAFE_FEATURE_SET_NAME
    assert config_module.SAFE_PLUGIN_NAME is config_module.SAFE_IDENTIFIER
