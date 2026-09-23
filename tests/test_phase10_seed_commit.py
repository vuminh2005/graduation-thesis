"""Phase 10 follow-up: the searcher seed follows training.seed; mltool_commit."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pandas as pd
import pytest

import mltool.build_info as build_info
from mltool.autogluon_adapter import AutoGluonAdapter, AutoGluonError, searcher_seed
from mltool.cli import register_project
from mltool.config import HpoConfig, ModelConfig, TaskConfig, TrainingConfig, load_config
from mltool.experiment import build_experiment_plan
from mltool.finalize import FinalizeError, load_finalize_input
from mltool.training import load_persisted_leaderboard
from mltool.tuning import TuningError, build_tuning_selection, load_persisted_tuning, recorded_hpo
from test_audit_fixes import build, edit, fake_adapters  # noqa: F401  (autouse fixture)
from test_phase4 import FakePredictor
from test_phase7 import by_phase

REPO = Path(__file__).resolve().parents[1]


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd, capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture(autouse=True)
def fresh_commit_cache() -> Any:
    build_info.mltool_commit.cache_clear()
    yield
    build_info.mltool_commit.cache_clear()


# =========================== searcher seed =======================================


@pytest.mark.parametrize(
    ("searcher", "seed", "expected"),
    [("random", 42, 42), ("random", 7, 7), ("random", None, 0), ("grid", 42, None), ("grid", None, None)],
)
def test_searcher_seed_derivation(searcher: str, seed: int | None, expected: int | None) -> None:
    assert searcher_seed(HpoConfig(2, 4, 30, searcher), TrainingConfig(seed=seed)) == expected


def tune_kwargs(tmp_path: Path, model: ModelConfig, hpo: HpoConfig, seed: int | None) -> dict[str, Any]:
    FakePredictor.fit_kwargs = {}
    adapter = AutoGluonAdapter(predictor_factory=FakePredictor, version_resolver=lambda: "t")
    train = pd.DataFrame({"x": [1, 2, 3], "label": ["no", "yes", "no"]})
    try:
        adapter.fit_predict(
            train_data=train, validation_features=pd.DataFrame({"x": [4, 5]}),
            task=TaskConfig("binary", "label", "yes"), model=model, primary_metric="roc_auc",
            predictor_path=tmp_path / "p", training=TrainingConfig(seed=seed), hpo=hpo,
        )
    except AutoGluonError:
        pass  # FakePredictor has no HPO outputs; the fit kwargs are already captured
    return FakePredictor.fit_kwargs["hyperparameter_tune_kwargs"]


def test_the_seed_reaches_autogluon_through_search_options(tmp_path: Path) -> None:
    kwargs = tune_kwargs(tmp_path, ModelConfig("g", "GBM"), HpoConfig(2, 4, 30), 42)
    assert kwargs["search_options"] == {"random_seed": 42}


def test_every_candidate_gets_the_same_searcher_seed(tmp_path: Path) -> None:
    seen = [
        tune_kwargs(tmp_path / str(i), model, HpoConfig(2, 4, 30), 42)["search_options"]
        for i, model in enumerate([ModelConfig("g", "GBM"), ModelConfig("c", "CAT"),
                                   ModelConfig("r", "RF", search_space={
                                       "max_depth": {"type": "int", "low": 2, "high": 8}})])
    ]
    assert seen == [{"random_seed": 42}] * 3


def test_a_null_seed_passes_nothing_exactly_as_before(tmp_path: Path) -> None:
    kwargs = tune_kwargs(tmp_path, ModelConfig("g", "GBM"), HpoConfig(2, 4, 30), None)
    assert kwargs == {"num_trials": 4, "scheduler": "local", "searcher": "random"}


def test_grid_gets_no_seed(tmp_path: Path) -> None:
    kwargs = tune_kwargs(tmp_path, ModelConfig("g", "GBM"), HpoConfig(2, 4, 30, "grid"), 42)
    assert "search_options" not in kwargs and kwargs["searcher"] == "local_grid"


def test_the_searcher_seed_is_recorded_everywhere(tmp_path: Path) -> None:
    path = build(tmp_path, through="finalize")  # training.seed 7
    assert register_project(path) == 0
    root = tmp_path
    manifest = json.loads((root / ".mltool/tuning/manifest.json").read_text())
    assert manifest["hpo"]["searcher_seed"] == 7
    for cid in manifest["candidate_ids"]:
        result = json.loads((root / ".mltool/tuning/candidates" / cid / "result.json").read_text())
        assert result["hpo"]["searcher_seed"] == 7, cid
    selected = json.loads((root / ".mltool/tuning/selected.json").read_text())
    final = json.loads((root / ".mltool/final/result.json").read_text())
    final_manifest = json.loads((root / ".mltool/final/manifest.json").read_text())
    registry = json.loads((root / ".mltool/registry/1/metadata.json").read_text())
    assert selected["searcher_seed"] == final["searcher_seed"] == 7
    assert final_manifest["selected"]["searcher_seed"] == registry["selected"]["searcher_seed"] == 7
    assert {r.data.params["searcher_seed"] for r in by_phase(root, "tune")} == {"7"}
    assert by_phase(root, "final")[0].data.params["searcher_seed"] == "7"


def test_a_seed_change_makes_train_tune_and_finalize_stale(tmp_path: Path) -> None:
    path = build(tmp_path, through="tune")
    edit(path, lambda raw: raw["training"].__setitem__("seed", 8))
    config = load_config(path)
    assert "training.seed" in load_persisted_leaderboard(config).warning
    with pytest.raises(TuningError, match='"training.seed" differs.*mltool train'):
        build_tuning_selection(build_experiment_plan(config))
    assert load_persisted_tuning(config).warning is not None
    with pytest.raises(FinalizeError, match='"training.seed" differs.*mltool train'):
        load_finalize_input(build_experiment_plan(config))


def test_records_from_before_the_seed_existed_read_as_seed_zero() -> None:
    old = {"top_n": 2, "num_trials": 4, "time_limit_seconds": 30, "scheduler": "local"}
    assert recorded_hpo({**old, "searcher": "random"})["searcher_seed"] == 0
    assert recorded_hpo({**old, "searcher": "grid"})["searcher_seed"] is None
    current = {**old, "searcher": "random", "searcher_seed": 42}
    assert recorded_hpo(current) == current


def test_an_old_manifest_is_stale_once_a_seed_would_change_the_sampling(tmp_path: Path) -> None:
    # The run below uses training.seed 7; an artifact from before the searcher seed
    # existed was sampled with AutoGluon's seed 0, so it no longer matches.
    path = build(tmp_path, through="tune")
    manifest_path = tmp_path / ".mltool/tuning/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["hpo"]["searcher_seed"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_persisted_tuning(load_config(path)).warning is not None


# =========================== mltool_commit =======================================


def test_mltool_commit_is_this_checkouts_head() -> None:
    record = build_info.mltool_commit()
    assert record is not None and record["commit"] == git("rev-parse", "HEAD", cwd=REPO)
    assert isinstance(record["dirty"], bool)


def test_source_commit_outside_git_is_null(tmp_path: Path) -> None:
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    assert build_info.source_commit(tmp_path) is None
    assert build_info.source_commit(tmp_path / "missing") is None


def test_an_untracked_package_inside_a_repo_does_not_borrow_its_head(tmp_path: Path) -> None:
    git("init", "-q", cwd=tmp_path)
    (tmp_path / "README").write_text("x", encoding="utf-8")
    git("add", "README", cwd=tmp_path)
    git("commit", "-qm", "c", cwd=tmp_path)
    package = tmp_path / ".venv/site-packages/mltool"  # like a non-editable install
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    assert build_info.source_commit(package) is None


def test_a_tracked_package_reports_its_commit_and_dirty_flag(tmp_path: Path) -> None:
    git("init", "-q", cwd=tmp_path)
    package = tmp_path / "src/mltool"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    git("add", ".", cwd=tmp_path)
    git("commit", "-qm", "c", cwd=tmp_path)
    head = git("rev-parse", "HEAD", cwd=tmp_path)
    assert build_info.source_commit(package) == {"commit": head, "dirty": False}
    (tmp_path / "notes.txt").write_text("changed", encoding="utf-8")  # outside the package
    assert build_info.source_commit(package) == {"commit": head, "dirty": False}
    (package / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    assert build_info.source_commit(package) == {"commit": head, "dirty": True}


def test_no_git_binary_is_null_not_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("git")

    monkeypatch.setattr(build_info.subprocess, "run", missing)
    assert build_info.source_commit(REPO / "src/mltool") is None


def test_every_manifest_and_the_registry_record_mltool_commit(tmp_path: Path) -> None:
    path = build(tmp_path, through="finalize")
    assert register_project(path) == 0
    expected = build_info.mltool_commit()
    for rel in ("training/manifest.json", "tuning/manifest.json", "final/manifest.json",
                "registry/1/metadata.json"):
        assert json.loads((tmp_path / ".mltool" / rel).read_text())["mltool_commit"] == expected, rel
    metadata = json.loads((tmp_path / ".mltool/registry/1/metadata.json").read_text())
    assert "git_commit" in metadata  # the project's commit, unchanged


def test_mltool_commit_is_null_when_not_running_from_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build_info, "PACKAGE_DIR", tmp_path / "not-a-checkout")
    path = build(tmp_path / "project", through="finalize")
    assert register_project(path) == 0
    for rel in ("training/manifest.json", "tuning/manifest.json", "final/manifest.json",
                "registry/1/metadata.json"):
        record = json.loads((tmp_path / "project/.mltool" / rel).read_text())
        assert "mltool_commit" in record and record["mltool_commit"] is None, rel
