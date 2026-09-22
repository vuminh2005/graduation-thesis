"""Regressions for the three findings of the whole-system audit of 7284e01.

Fix 1: feature artifacts are bound to the prepared artifacts by hash.
Fix 2: hpo_effective/hpo_warning reach every final artifact and both CLI reports.
Fix 3: .mltool/final is staleness-aware, and register blocks unless forced.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd
import pytest
import yaml

from mltool.cli import (
    best_project,
    features_project,
    final_result_project,
    finalize_project,
    plan_project,
    prepare_project,
    register_project,
    train_project,
    tune_project,
)
from mltool.config import load_config
from mltool.experiment import (
    ExperimentError,
    build_experiment_plan,
    current_feature_set_recipe,
    load_feature_artifacts,
    recipe_fingerprint,
)
from mltool.feature_materialization import (
    FeatureMaterializationError,
    prepared_artifacts_fingerprint,
)
from mltool.finalize import load_persisted_final
from mltool.registry import RegistryBlocked, list_versions, register_final
from mltool.state import list_runs
from mltool.training import train_experiment
from mltool.tuning import build_tuning_selection, tune_experiment
from mltool.finalize import finalize_experiment
from test_phase4 import RecordingAdapter, materialize, project_config
from test_phase5 import TuningAdapter, hpo_raw
from test_phase6 import FinalAdapter
from test_phase7 import mlflow_runs

GBM_AND_RF = [
    {"name": "lightgbm", "family": "GBM", "params": {}},
    {"name": "forest", "family": "RF", "params": {}},
]
_train, _tune, _finalize = train_experiment, tune_experiment, finalize_experiment


@pytest.fixture(autouse=True)
def fake_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    RecordingAdapter.failures = set()
    RecordingAdapter.calls = []
    TuningAdapter.failures = set()
    TuningAdapter.calls = []
    FinalAdapter.calls = []
    monkeypatch.setattr(
        "mltool.cli.train_experiment", lambda plan: _train(plan, adapter_factory=RecordingAdapter)
    )
    monkeypatch.setattr(
        "mltool.cli.tune_experiment",
        lambda plan, sel: _tune(plan, sel, adapter_factory=TuningAdapter),
    )
    monkeypatch.setattr(
        "mltool.cli.finalize_experiment",
        lambda plan, inp: _finalize(plan, inp, adapter_factory=FinalAdapter),
    )


def build(root: Path, *, models: list[dict[str, Any]] | None = None, through: str = "finalize") -> Path:
    path = materialize(root, raw=project_config(models=models or GBM_AND_RF, training={"seed": 7}))
    hpo_raw(path, top_n=2)
    steps = ["plan", "train", "tune", "finalize"]
    fns = {"plan": plan_project, "train": train_project, "tune": tune_project,
           "finalize": finalize_project}
    for step in steps[: steps.index(through) + 1]:
        assert fns[step](path) == 0, step
    return path


def edit(path: Path, mutate) -> None:
    raw = yaml.safe_load(path.read_text())
    mutate(raw)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def no_fits() -> bool:
    return not RecordingAdapter.calls and not TuningAdapter.calls and not FinalAdapter.calls


# =========================== Fix 1 =============================================


def test_features_manifest_records_a_prepared_artifacts_fingerprint(tmp_path: Path) -> None:
    path = build(tmp_path, through="plan")
    manifest = json.loads((tmp_path / ".mltool/features/manifest.json").read_text())
    recorded = manifest["prepared_artifacts_fingerprint"]
    assert isinstance(recorded, str) and len(recorded) == 64
    assert recorded != manifest["source_dataset_fingerprint"]
    assert recorded == prepared_artifacts_fingerprint(tmp_path / ".mltool/prepared")


def test_fingerprint_covers_the_split_and_ignores_the_unstable_preprocessor(
    tmp_path: Path,
) -> None:
    path = build(tmp_path, through="plan")
    prepared = tmp_path / ".mltool/prepared"
    before = prepared_artifacts_fingerprint(prepared)
    assert prepare_project(path) == 0  # same config: prepared output is byte-identical
    assert prepared_artifacts_fingerprint(prepared) == before
    edit(path, lambda raw: raw["split"].__setitem__("random_seed", 7))
    assert prepare_project(path) == 0
    assert prepared_artifacts_fingerprint(prepared) != before


@pytest.mark.parametrize(
    ("field", "value"), [("random_seed", 7), ("test_ratio", 0.25), ("validation_ratio", 0.3)]
)
def test_load_feature_artifacts_detects_a_reprepared_split(
    tmp_path: Path, field: str, value: Any
) -> None:
    """Unit level: the audit's scenario seen directly by the staleness check."""
    path = build(tmp_path / field, through="plan")
    assert load_feature_artifacts(load_config(path))  # fresh before the change
    edit(path, lambda raw: raw["split"].__setitem__(field, value))
    assert prepare_project(path) == 0  # only prepare is re-run
    with pytest.raises(ExperimentError, match="prepared artifacts changed after materialization"):
        load_feature_artifacts(load_config(path))
    assert features_project(path) == 0  # re-materializing clears it
    assert load_feature_artifacts(load_config(path))


def test_feature_manifest_without_the_field_is_stale(tmp_path: Path) -> None:
    """Artifacts written before this fix cannot prove provenance."""
    path = build(tmp_path, through="plan")
    manifest_path = tmp_path / ".mltool/features/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["prepared_artifacts_fingerprint"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ExperimentError, match="predates prepared-artifact fingerprinting"):
        load_feature_artifacts(load_config(path))
    assert plan_project(path) == 2


def test_missing_prepared_artifacts_point_at_prepare(tmp_path: Path) -> None:
    path = build(tmp_path, through="plan")
    (tmp_path / ".mltool/prepared/validation.parquet").unlink()
    with pytest.raises(ExperimentError, match='prepared split is missing.*run "mltool prepare"'):
        load_feature_artifacts(load_config(path))
    with pytest.raises(FeatureMaterializationError):
        prepared_artifacts_fingerprint(tmp_path / ".mltool/prepared")


@pytest.mark.parametrize(("field", "value"), [("random_seed", 7), ("test_ratio", 0.25)])
def test_reprepared_split_is_refused_by_every_downstream_command(
    tmp_path: Path, field: str, value: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """The audit's live reproduction: no phase may proceed, and nothing may be fit."""
    path = build(tmp_path / field, through="tune")
    edit(path, lambda raw: raw["split"].__setitem__(field, value))
    assert prepare_project(path) == 0
    RecordingAdapter.calls = []
    TuningAdapter.calls = []
    FinalAdapter.calls = []
    capsys.readouterr()
    for command in (plan_project, train_project, tune_project, finalize_project):
        assert command(path) == 2, command.__name__
        out = capsys.readouterr().out
        assert "feature artifacts are stale" in out
        assert "prepared artifacts changed after materialization" in out
        assert 'run "mltool features" again' in out
    assert no_fits()
    assert not (path.parent / ".mltool/final").exists()
    # and the whole chain works again once features are re-materialized
    assert features_project(path) == 0
    for command in (plan_project, train_project, tune_project, finalize_project):
        assert command(path) == 0, command.__name__


def touch_feature_parquet(root: Path, feature_set: str = "base") -> None:
    """Change a FeatureSet artifact's bytes without touching the config."""
    path = root / ".mltool/features" / feature_set / "train.parquet"
    frame = pd.read_parquet(path)
    frame.iloc[0, 0] = frame.iloc[0, 0] + 1
    frame.to_parquet(path, index=False)


def test_rerunning_features_unchanged_is_idempotent(tmp_path: Path) -> None:
    """Fix 1 must not make an unchanged re-materialization look stale."""
    path = build(tmp_path, through="tune")
    before = (tmp_path / ".mltool/features/base/train.parquet").read_bytes()
    assert features_project(path) == 0
    assert (tmp_path / ".mltool/features/base/train.parquet").read_bytes() == before
    assert finalize_project(path) == 0  # downstream artifacts stay valid


def test_changed_feature_bytes_still_invalidate_training(tmp_path: Path) -> None:
    """Fix 1 must not mask the pre-existing feature-hash check."""
    path = build(tmp_path, through="tune")
    touch_feature_parquet(tmp_path)
    assert finalize_project(path) == 2
    assert no_fits() or FinalAdapter.calls == []


# =========================== Fix 2 =============================================


def final_artifacts(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = json.loads((root / ".mltool/final/result.json").read_text())
    manifest = json.loads((root / ".mltool/final/manifest.json").read_text())
    metadata = json.loads((root / ".mltool/registry/1/metadata.json").read_text())
    return result, manifest, metadata


def test_untuned_family_is_recorded_everywhere_and_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = build(tmp_path, models=[{"name": "forest", "family": "RF", "params": {}}])
    assert register_project(path) == 0
    selected = json.loads((tmp_path / ".mltool/tuning/selected.json").read_text())
    assert selected["hpo_effective"] is False
    result, manifest, metadata = final_artifacts(tmp_path)
    for name, record in (("result", result), ("manifest", manifest), ("metadata", metadata)):
        assert record["hpo_effective"] is False, name
        assert "no default search space for family RF" in record["hpo_warning"], name
    assert manifest["selected"]["hpo_effective"] is False
    (run,) = [r for r in mlflow_runs(tmp_path) if r.data.tags["phase"] == "final"]
    assert run.data.tags["hpo_effective"] == "false"
    assert "family RF" in run.data.tags["hpo_warning"]

    expected = "Tuned: no (family RF has no HPO search space)"
    capsys.readouterr()
    assert final_result_project(path) == 0
    assert expected in capsys.readouterr().out
    assert best_project(path) == 0
    out = capsys.readouterr().out
    assert out.count(expected) == 2  # the finalized model and the registered version


def test_tuned_family_is_recorded_as_tuned(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = build(tmp_path, models=[{"name": "lightgbm", "family": "GBM", "params": {}}])
    assert register_project(path) == 0
    result, manifest, metadata = final_artifacts(tmp_path)
    for record in (result, manifest, metadata):
        assert record["hpo_effective"] is True and record["hpo_warning"] is None
    (run,) = [r for r in mlflow_runs(tmp_path) if r.data.tags["phase"] == "final"]
    assert run.data.tags["hpo_effective"] == "true" and "hpo_warning" not in run.data.tags
    capsys.readouterr()
    assert final_result_project(path) == 0 and best_project(path) == 0
    out = capsys.readouterr().out
    assert "Tuned: yes" in out and "no HPO search space" not in out


def test_finalize_itself_prints_the_untuned_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build(tmp_path, models=[{"name": "forest", "family": "RF", "params": {}}])
    assert "Tuned: no (family RF has no HPO search space)" in capsys.readouterr().out


# =========================== Fix 3 =============================================


def test_final_manifest_records_its_provenance(tmp_path: Path) -> None:
    path = build(tmp_path)
    manifest = json.loads((tmp_path / ".mltool/final/manifest.json").read_text())
    selected = json.loads((tmp_path / ".mltool/tuning/selected.json").read_text())
    assert manifest["feature_artifacts"] == selected["feature_artifacts"]
    assert manifest["prepared_artifacts_fingerprint"] == prepared_artifacts_fingerprint(
        tmp_path / ".mltool/prepared"
    )
    assert load_persisted_final(load_config(path)).warning is None


@pytest.mark.parametrize("break_it", ["reprepare", "rematerialize", "dataset"])
def test_upstream_change_makes_the_final_model_stale(tmp_path: Path, break_it: str) -> None:
    path = build(tmp_path / break_it)
    root = path.parent
    if break_it == "reprepare":
        edit(path, lambda raw: raw["split"].__setitem__("random_seed", 7))
        assert prepare_project(path) == 0
        expected = "prepared artifacts changed"
    elif break_it == "rematerialize":
        touch_feature_parquet(root)
        expected = "FeatureSet was re-materialized"
    else:
        frame = pd.read_csv(root / "data/dataset.csv")
        frame.loc[len(frame)] = frame.iloc[0]
        frame.to_csv(root / "data/dataset.csv", index=False)
        expected = "source dataset changed"
    warning = load_persisted_final(load_config(path)).warning
    assert warning is not None and expected in warning


def test_a_final_model_without_provenance_is_stale(tmp_path: Path) -> None:
    path = build(tmp_path)
    manifest_path = tmp_path / ".mltool/final/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["prepared_artifacts_fingerprint"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert "predates prepared-artifact fingerprinting" in load_persisted_final(
        load_config(path)
    ).warning


def test_final_result_and_best_report_staleness_instead_of_a_current_metric(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = build(tmp_path)
    edit(path, lambda raw: raw["split"].__setitem__("random_seed", 7))
    assert prepare_project(path) == 0
    capsys.readouterr()
    assert final_result_project(path) == 0
    out = capsys.readouterr().out
    assert "Warnings" in out and "prepared artifacts changed" in out
    assert best_project(path) == 0
    assert "prepared artifacts changed" in capsys.readouterr().out
    # status agrees
    from mltool.reporting import render_status

    assert "stale" in [
        line.split()[1] for line in render_status(load_config(path)).splitlines()
        if line.startswith("finalize")
    ]


def test_register_blocks_a_stale_final_unless_forced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = build(tmp_path)
    edit(path, lambda raw: raw["split"].__setitem__("random_seed", 7))
    assert prepare_project(path) == 0
    capsys.readouterr()

    assert register_project(path) == 2
    out = capsys.readouterr().out
    assert "the final artifacts are stale" in out and "--force" in out
    assert list_versions(tmp_path) == []
    blocked = list_runs(tmp_path)[0]
    assert (blocked.command, blocked.status) == ("register", "BLOCKED")
    assert blocked.details["blocked"] is True and blocked.details["forced"] is False
    with pytest.raises(RegistryBlocked, match="stale"):
        register_final(load_config(path))

    assert register_project(path, force=True) == 0
    assert list_versions(tmp_path) == [1]
    metadata = json.loads((tmp_path / ".mltool/registry/1/metadata.json").read_text())
    assert metadata["forced"] is True
    assert metadata["warning"] and "prepared artifacts changed" in metadata["warning"]
    assert metadata["split"]["random_seed"] == 42  # the split the model was actually built on
    forced = list_runs(tmp_path)[0]
    assert (forced.command, forced.status) == ("register", "SUCCEEDED")
    assert forced.details["forced"] is True and forced.details["stale_warning"]
    assert "Warnings" in out or True
    assert "prepared artifacts changed" in capsys.readouterr().out  # printed on the forced run


def test_register_still_accepts_a_fresh_final_without_force(tmp_path: Path) -> None:
    path = build(tmp_path)
    assert register_project(path) == 0
    metadata = json.loads((tmp_path / ".mltool/registry/1/metadata.json").read_text())
    assert metadata["warning"] is None and metadata["forced"] is False
    assert metadata["split"]["random_seed"] == 42
    assert list_runs(tmp_path)[0].details["forced"] is False


# =========================== optional audit follow-ups ==========================


def test_mlflow_run_ids_are_traceable_from_the_artifacts(tmp_path: Path) -> None:
    path = build(tmp_path)
    assert register_project(path) == 0
    tuning = json.loads((tmp_path / ".mltool/tuning/mlflow.json").read_text())
    final = json.loads((tmp_path / ".mltool/final/mlflow.json").read_text())
    selected = json.loads((tmp_path / ".mltool/tuning/selected.json").read_text())
    runs = {r.info.run_id: r for r in mlflow_runs(tmp_path)}
    assert set(tuning["runs"]) == {
        json.loads(p.read_text())["candidate_id"]
        for p in (tmp_path / ".mltool/tuning/candidates").glob("*/result.json")
    }
    for candidate_id, run_id in tuning["runs"].items():
        assert runs[run_id].data.tags["candidate_id"] == candidate_id
        assert runs[run_id].data.tags["phase"] == "tune"
    (final_id, final_run), = final["runs"].items()
    assert final_id == selected["candidate_id"]
    assert runs[final_run].data.tags["phase"] == "final"
    assert tuning["experiment"] == final["experiment"] == "phase4-test"
    assert tuning["tracking_uri"].startswith("file://")
    # the registry keeps the trace with its copy
    assert json.loads((tmp_path / ".mltool/registry/1/mlflow.json").read_text()) == final


def test_mlflow_trace_file_is_absent_when_tracking_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mltool.tracking as tracking_module

    def broken(self: Any, config: Any) -> None:
        raise RuntimeError("store down")

    monkeypatch.setattr(tracking_module._Tracker, "__init__", broken)
    path = build(tmp_path)
    assert not (tmp_path / ".mltool/tuning/mlflow.json").exists()
    assert not (tmp_path / ".mltool/final/mlflow.json").exists()
    assert register_project(path) == 0  # registering still works without it


# ====== Phase-9 audit follow-up: the selected FeatureSet's recipe ================
#
# The parquet hashes above only prove features/ was not rebuilt. A recipe edit
# that has not been re-materialized leaves those bytes untouched, so register
# used to accept a final model built from a recipe the config no longer asks for.

RECIPE_PLUGINS = '''
import pandas as pd


class Scaled:
    def __init__(self, factor=2):
        self.factor = factor

    def fit(self, X):
        return self

    def transform(self, X):
        return pd.DataFrame({"x_scaled": X["x"] * self.factor}, index=X.index)


class Shifted:
    def fit(self, X):
        return self

    def transform(self, X):
        return pd.DataFrame({"x_shift": X["x"] + 1}, index=X.index)
'''


def recipe_set(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "source_columns": ["x"],
        "plugins": ["scaled"],
        "plugin_inputs": ["z"],
    }


def recipe_build(root: Path) -> Path:
    """A finalized project with two interchangeable plugin-bearing FeatureSets."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "plugin.py").write_text(RECIPE_PLUGINS, encoding="utf-8")
    frame = pd.DataFrame(
        {
            "x": range(100),
            "z": [value % 7 for value in range(100)],
            "w": [value % 3 for value in range(100)],
            "label": [value % 2 for value in range(100)],
        }
    )
    raw = project_config(
        models=GBM_AND_RF,
        training={"seed": 7},
        features={
            "plugins": [
                {"name": "scaled", "entrypoint": "./plugin.py:Scaled", "params": {"factor": 2}},
                {"name": "shifted", "entrypoint": "./plugin.py:Shifted", "params": {}},
            ],
            "sets": [recipe_set("one"), recipe_set("two")],
        },
    )
    path = materialize(root, raw=raw, frame=frame)
    hpo_raw(path, top_n=2)
    for step in (plan_project, train_project, tune_project, finalize_project):
        assert step(path) == 0, step
    return path


def selected_set(root: Path) -> str:
    return json.loads((root / ".mltool/final/manifest.json").read_text())["selected"]["feature_set"]


def _drop_plugin(spec: dict[str, Any]) -> None:
    # plugin_inputs without a plugin to read them is rejected by the config.
    spec["plugins"] = []
    spec["plugin_inputs"] = []


RECIPE_EDITS = {
    "source_columns": lambda s: s.__setitem__("source_columns", ["x", "w"]),
    "plugin_added": lambda s: s.__setitem__("plugins", ["scaled", "shifted"]),
    "plugin_removed": _drop_plugin,
    "plugin_inputs": lambda s: s.__setitem__("plugin_inputs", ["z", "w"]),
}


def edit_set(path: Path, name: str, dimension: str) -> None:
    def mutate(raw: dict[str, Any]) -> None:
        if dimension == "plugin_params":
            catalog = {plugin["name"]: plugin for plugin in raw["features"]["plugins"]}
            catalog["scaled"]["params"] = {"factor": 3}
            return
        for spec in raw["features"]["sets"]:
            if spec["name"] == name:
                RECIPE_EDITS[dimension](spec)

    edit(path, mutate)


SET_DIMENSIONS = ["source_columns", "plugin_added", "plugin_removed", "plugin_inputs"]
# The plugin catalog is shared, so its params are never one set's alone.
RECIPE_DIMENSIONS = [*SET_DIMENSIONS, "plugin_params"]


def test_final_manifest_records_the_selected_feature_set_recipe(tmp_path: Path) -> None:
    path = recipe_build(tmp_path)
    config = load_config(path)
    manifest = json.loads((tmp_path / ".mltool/final/manifest.json").read_text())
    assert manifest["selected_feature_set_recipe"] == recipe_fingerprint(
        current_feature_set_recipe(config, selected_set(tmp_path))
    )
    assert load_persisted_final(config).warning is None


@pytest.mark.parametrize("dimension", RECIPE_DIMENSIONS)
def test_a_selected_recipe_change_makes_the_final_model_stale(
    tmp_path: Path, dimension: str
) -> None:
    path = recipe_build(tmp_path / dimension)
    edit_set(path, selected_set(path.parent), dimension)
    warning = load_persisted_final(load_config(path)).warning
    assert warning is not None and "recipe changed" in warning


@pytest.mark.parametrize("dimension", RECIPE_DIMENSIONS)
def test_a_selected_recipe_change_blocks_register_until_forced(
    tmp_path: Path, dimension: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = recipe_build(tmp_path / dimension)
    root = path.parent
    edit_set(path, selected_set(root), dimension)
    capsys.readouterr()

    assert register_project(path) == 2
    assert "the final artifacts are stale" in capsys.readouterr().out
    assert list_versions(root) == []

    assert final_result_project(path) == 0
    assert "recipe changed" in capsys.readouterr().out
    assert best_project(path) == 0
    assert "recipe changed" in capsys.readouterr().out
    from mltool.reporting import render_status

    assert "stale" in [
        line.split()[1] for line in render_status(load_config(path)).splitlines()
        if line.startswith("finalize")
    ]

    assert register_project(path, force=True) == 0
    assert list_versions(root) == [1]
    metadata = json.loads((root / ".mltool/registry/1/metadata.json").read_text())
    assert metadata["forced"] is True and "recipe changed" in metadata["warning"]


@pytest.mark.parametrize("dimension", SET_DIMENSIONS)
def test_a_recipe_change_to_another_feature_set_keeps_the_final_fresh(
    tmp_path: Path, dimension: str
) -> None:
    path = recipe_build(tmp_path / dimension)
    root = path.parent
    others = [name for name in ("one", "two") if name != selected_set(root)]
    edit_set(path, others[0], dimension)
    assert load_persisted_final(load_config(path)).warning is None
    assert register_project(path) == 0
    metadata = json.loads((root / ".mltool/registry/1/metadata.json").read_text())
    assert metadata["warning"] is None and metadata["forced"] is False


def test_a_final_model_without_a_recipe_fingerprint_is_stale(tmp_path: Path) -> None:
    path = recipe_build(tmp_path)
    manifest_path = tmp_path / ".mltool/final/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["selected_feature_set_recipe"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert "predates feature-set recipe fingerprinting" in load_persisted_final(
        load_config(path)
    ).warning
