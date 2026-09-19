from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from mltool.cli import features_project, prepare_project, validate_project
from mltool.config import ConfigError, load_config


MISSING = object()


def source_frame(rows: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": range(rows),
            "age": [20 + index % 30 for index in range(rows)],
            "income": [1000.0 + index * 10 for index in range(rows)],
            "category": [f"group-{index % 3}" for index in range(rows)],
            "label": [index % 2 for index in range(rows)],
        }
    )


def write_project(
    root: Path,
    *,
    features: object = MISSING,
    frame: pd.DataFrame | None = None,
) -> Path:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (frame if frame is not None else source_frame()).to_csv(
        data_dir / "dataset.csv", index=False
    )
    config: dict[str, object] = {
        "schema_version": "0.1",
        "project": {"name": "phase-3-test"},
        "task": {"type": "binary", "target": "label", "positive_class": 1},
        "data": {"format": "auto", "path": "./data/dataset.csv"},
        "validation": {"enabled": True, "fail_on_error": True},
        "split": {
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "stratify": "auto",
            "random_seed": 42,
        },
        "preprocessing": {
            "external": {"enabled": False, "entrypoint": None, "params": {}}
        },
    }
    if features is not MISSING:
        config["features"] = features
    path = root / "mltool.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def prepare(config_path: Path) -> None:
    assert prepare_project(config_path) == 0


def read_set(root: Path, set_name: str, split: str = "train") -> pd.DataFrame:
    return pd.read_parquet(root / f".mltool/features/{set_name}/{split}.parquet")


def base_features_config() -> dict[str, object]:
    return {
        "plugins": [],
        "sets": [{"name": "base", "source_columns": ["*"], "plugins": []}],
    }


def test_missing_features_section_defaults_to_base_and_materializes(
    tmp_path: Path,
) -> None:
    config_path = write_project(tmp_path)
    config = load_config(config_path)
    assert [item.name for item in config.features.sets] == ["base"]
    assert config.features.sets[0].source_columns == ["*"]
    prepare(config_path)
    prepared_before = {
        split: (tmp_path / f".mltool/prepared/{split}.parquet").read_bytes()
        for split in ("train", "validation", "test")
    }

    assert features_project(config_path) == 0

    for split in ("train", "validation", "test"):
        frame = read_set(tmp_path, "base", split)
        assert frame.columns.tolist() == [
            "row_id",
            "age",
            "income",
            "category",
            "label",
        ]
    assert (tmp_path / ".mltool/features/base/manifest.json").is_file()
    assert (tmp_path / ".mltool/features/manifest.json").is_file()
    for split, contents in prepared_before.items():
        assert (tmp_path / f".mltool/prepared/{split}.parquet").read_bytes() == contents


def test_explicit_source_subset_preserves_yaml_order(tmp_path: Path) -> None:
    features = {
        "plugins": [],
        "sets": [
            {
                "name": "numeric",
                "source_columns": ["income", "age"],
                "plugins": [],
            }
        ],
    }
    config_path = write_project(tmp_path, features=features)
    prepare(config_path)

    assert features_project(config_path) == 0
    assert read_set(tmp_path, "numeric").columns.tolist() == [
        "income",
        "age",
        "label",
    ]


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (
            {
                "plugins": [],
                "sets": [
                    {
                        "name": "bad",
                        "source_columns": ["*", "age"],
                        "plugins": [],
                    }
                ],
            },
            '"*" must appear alone',
        ),
        (
            {
                "plugins": [],
                "sets": [
                    {"name": "same", "source_columns": ["*"], "plugins": []},
                    {"name": "same", "source_columns": ["age"], "plugins": []},
                ],
            },
            "duplicate feature-set name",
        ),
        (
            {
                "plugins": [
                    {"name": "same", "entrypoint": "a.py:A", "params": {}},
                    {"name": "same", "entrypoint": "b.py:B", "params": {}},
                ],
                "sets": [{"name": "base", "source_columns": ["*"], "plugins": []}],
            },
            "duplicate feature plugin name",
        ),
        (
            {
                "plugins": [],
                "sets": [
                    {"name": "bad", "source_columns": ["*"], "plugins": ["unknown"]}
                ],
            },
            "references unknown plugin",
        ),
        (
            {
                "plugins": [],
                "sets": [
                    {"name": "../../escape", "source_columns": ["*"], "plugins": []}
                ],
            },
            "is unsafe",
        ),
        (
            {
                "plugins": [],
                "sets": [
                    {
                        "name": "bad",
                        "source_columns": ["age", "age"],
                        "plugins": [],
                    }
                ],
            },
            "source_columns.*unique",
        ),
        ({"plugins": [], "sets": []}, "sets.*non-empty"),
    ],
)
def test_invalid_feature_configuration_is_rejected(
    tmp_path: Path, features: dict[str, object], message: str
) -> None:
    config_path = write_project(tmp_path, features=features)
    with pytest.raises(ConfigError, match=message):
        load_config(config_path)


def test_missing_source_column_and_target_selection_fail_clearly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    features = {
        "plugins": [],
        "sets": [
            {"name": "missing", "source_columns": ["not_there"], "plugins": []}
        ],
    }
    config_path = write_project(tmp_path, features=features)
    prepare(config_path)
    assert features_project(config_path) == 2
    assert 'requests missing source column "not_there"' in capsys.readouterr().out

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["features"]["sets"][0]["source_columns"] = ["label"]
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    assert features_project(config_path) == 2
    assert "may not select configured target" in capsys.readouterr().out


AUDIT_PLUGIN = '''
import json
from pathlib import Path
from uuid import uuid4
import pandas as pd

class AuditFeatures:
    def __init__(self, multiplier=1, audit_file=None):
        self.multiplier = multiplier
        self.audit_file = audit_file
        self.token = uuid4().hex
        self.mean_ = None

    def _record(self, kind, X):
        path = Path(self.audit_file)
        events = json.loads(path.read_text()) if path.exists() else []
        events.append({
            "token": self.token,
            "kind": kind,
            "columns": list(X.columns),
            "row_ids": X["row_id"].tolist(),
        })
        path.write_text(json.dumps(events))

    def fit(self, X_train):
        self._record("fit", X_train)
        self.mean_ = X_train["income"].mean()

    def transform(self, X):
        self._record("transform", X)
        return pd.DataFrame({
            "centered_income": (X["income"] - self.mean_) * self.multiplier,
        }, index=X.index)
'''


def test_plugin_relative_entrypoint_train_only_state_and_independent_instances(
    tmp_path: Path,
) -> None:
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    (plugins_dir / "features.py").write_text(AUDIT_PLUGIN, encoding="utf-8")
    audit_path = tmp_path / "plugin-events.json"
    features = {
        "plugins": [
            {
                "name": "audit",
                "entrypoint": "./plugins/features.py:AuditFeatures",
                "params": {"multiplier": 2, "audit_file": str(audit_path)},
            }
        ],
        "sets": [
            {
                "name": "focused",
                "source_columns": ["row_id", "income"],
                "plugins": ["audit"],
            },
            {
                "name": "wide",
                "source_columns": ["row_id", "age", "income"],
                "plugins": ["audit"],
            },
        ],
    }
    config_path = write_project(tmp_path, features=features)
    prepare(config_path)

    assert features_project(config_path) == 0

    events = json.loads(audit_path.read_text(encoding="utf-8"))
    tokens = {event["token"] for event in events}
    assert len(tokens) == 2
    for token in tokens:
        instance_events = [event for event in events if event["token"] == token]
        assert [event["kind"] for event in instance_events] == [
            "fit",
            "transform",
            "transform",
            "transform",
        ]
        fit_event = instance_events[0]
        assert "label" not in fit_event["columns"]
        assert len(fit_event["row_ids"]) == 42
        assert instance_events[1]["row_ids"] == fit_event["row_ids"]
        assert not set(instance_events[2]["row_ids"]) & set(fit_event["row_ids"])
        assert not set(instance_events[3]["row_ids"]) & set(fit_event["row_ids"])

    for set_name in ("focused", "wide"):
        schemas: list[list[str]] = []
        for split in ("train", "validation", "test"):
            frame = read_set(tmp_path, set_name, split)
            schemas.append(frame.columns.tolist())
            assert frame["label"].tolist() == [
                row_id % 2 for row_id in frame["row_id"]
            ]
            train = read_set(tmp_path, set_name, "train")
            train_mean = train["income"].mean()
            expected = (frame["income"] - train_mean) * 2
            pd.testing.assert_series_equal(
                frame["centered_income"], expected, check_names=False
            )
        assert schemas[0] == schemas[1] == schemas[2]


@pytest.mark.parametrize(
    ("transform_body", "expected_error"),
    [
        ("return X.to_dict()", "must return a pandas DataFrame"),
        ("return X.iloc[:-1, :1].copy()", "changed row count"),
        ("return X.iloc[::-1, :1].copy()", "changed or reordered the index"),
        (
            'result = X.iloc[:, :2].copy()\n        result.columns = ["dup", "dup"]\n        return result',
            "generated duplicate columns",
        ),
        (
            'return pd.DataFrame({"label": 1}, index=X.index)',
            "generated configured target column",
        ),
        (
            'return pd.DataFrame({"age": 1}, index=X.index)',
            "generated colliding column",
        ),
        ("return pd.DataFrame(index=X.index)", "generated zero feature columns"),
    ],
)
def test_feature_plugin_output_contract_violations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    transform_body: str,
    expected_error: str,
) -> None:
    (tmp_path / "plugin.py").write_text(
        "import pandas as pd\n"
        "class BadFeatures:\n"
        "    def fit(self, X_train):\n"
        "        pass\n"
        "    def transform(self, X):\n"
        f"        {transform_body}\n",
        encoding="utf-8",
    )
    features = {
        "plugins": [
            {"name": "bad", "entrypoint": "./plugin.py:BadFeatures", "params": {}}
        ],
        "sets": [{"name": "bad_set", "source_columns": ["*"], "plugins": ["bad"]}],
    }
    config_path = write_project(tmp_path, features=features)
    prepare(config_path)

    assert features_project(config_path) == 2
    output = capsys.readouterr().out
    assert expected_error in output
    assert "Traceback" not in output
    assert not (tmp_path / ".mltool/features").exists()


def test_two_plugins_cannot_generate_same_column(tmp_path: Path) -> None:
    (tmp_path / "plugin.py").write_text(
        "import pandas as pd\n"
        "class Generated:\n"
        "    def __init__(self, value): self.value = value\n"
        "    def fit(self, X): pass\n"
        "    def transform(self, X):\n"
        "        return pd.DataFrame({'same': self.value}, index=X.index)\n",
        encoding="utf-8",
    )
    features = {
        "plugins": [
            {"name": "one", "entrypoint": "./plugin.py:Generated", "params": {"value": 1}},
            {"name": "two", "entrypoint": "./plugin.py:Generated", "params": {"value": 2}},
        ],
        "sets": [
            {"name": "collision", "source_columns": ["*"], "plugins": ["one", "two"]}
        ],
    }
    config_path = write_project(tmp_path, features=features)
    prepare(config_path)

    assert features_project(config_path) == 2


def test_plugin_outputs_are_independent_not_chained(tmp_path: Path) -> None:
    (tmp_path / "plugin.py").write_text(
        "import pandas as pd\n"
        "class A:\n"
        "    def fit(self, X): pass\n"
        "    def transform(self, X): return pd.DataFrame({'from_a': 1}, index=X.index)\n"
        "class B:\n"
        "    def fit(self, X):\n"
        "        if 'from_a' in X: raise RuntimeError('plugin chaining detected')\n"
        "    def transform(self, X): return pd.DataFrame({'from_b': 2}, index=X.index)\n",
        encoding="utf-8",
    )
    features = {
        "plugins": [
            {"name": "a", "entrypoint": "./plugin.py:A", "params": {}},
            {"name": "b", "entrypoint": "./plugin.py:B", "params": {}},
        ],
        "sets": [{"name": "both", "source_columns": ["age"], "plugins": ["a", "b"]}],
    }
    config_path = write_project(tmp_path, features=features)
    prepare(config_path)

    assert features_project(config_path) == 0
    assert read_set(tmp_path, "both").columns.tolist() == [
        "age",
        "from_a",
        "from_b",
        "label",
    ]


def test_plugin_schema_difference_across_splits_fails(tmp_path: Path) -> None:
    (tmp_path / "plugin.py").write_text(
        "import pandas as pd\n"
        "class Variable:\n"
        "    def fit(self, X): pass\n"
        "    def transform(self, X):\n"
        "        name = 'large_split' if len(X) > 10 else 'small_split'\n"
        "        return pd.DataFrame({name: 1}, index=X.index)\n",
        encoding="utf-8",
    )
    features = {
        "plugins": [{"name": "variable", "entrypoint": "./plugin.py:Variable", "params": {}}],
        "sets": [{"name": "bad", "source_columns": ["*"], "plugins": ["variable"]}],
    }
    config_path = write_project(tmp_path, features=features)
    prepare(config_path)
    assert features_project(config_path) == 2


@pytest.mark.parametrize(
    ("plugin_source", "class_name", "expected_error"),
    [
        ("class Existing: pass\n", "Missing", "class \"Missing\" was not found"),
        ("symbol = 3\n", "symbol", "is not a class"),
        (
            "class MissingFit:\n    def transform(self, X): return X\n",
            "MissingFit",
            "must provide callable fit",
        ),
        (
            "class MissingTransform:\n    def fit(self, X): pass\n",
            "MissingTransform",
            "must provide callable transform",
        ),
    ],
)
def test_feature_plugin_loading_contract_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    plugin_source: str,
    class_name: str,
    expected_error: str,
) -> None:
    (tmp_path / "plugin.py").write_text(plugin_source, encoding="utf-8")
    features = {
        "plugins": [
            {"name": "plugin", "entrypoint": f"./plugin.py:{class_name}", "params": {}}
        ],
        "sets": [{"name": "set", "source_columns": ["*"], "plugins": ["plugin"]}],
    }
    config_path = write_project(tmp_path, features=features)
    prepare(config_path)
    assert features_project(config_path) == 2
    assert expected_error in capsys.readouterr().out


def test_missing_plugin_file_and_bad_constructor_fail_with_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    features = {
        "plugins": [{"name": "missing", "entrypoint": "./missing.py:Missing", "params": {}}],
        "sets": [{"name": "set", "source_columns": ["*"], "plugins": ["missing"]}],
    }
    config_path = write_project(tmp_path, features=features)
    prepare(config_path)
    assert features_project(config_path) == 2
    assert "feature plugin file not found" in capsys.readouterr().out

    (tmp_path / "plugin.py").write_text(
        "class NeedsParam:\n"
        "    def __init__(self, required): pass\n"
        "    def fit(self, X): pass\n"
        "    def transform(self, X): return X.iloc[:, :1]\n",
        encoding="utf-8",
    )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["features"]["plugins"][0]["entrypoint"] = "./plugin.py:NeedsParam"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    assert features_project(config_path) == 2
    assert "could not instantiate feature plugin" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("fit_body", "transform_body", "expected_error"),
    [
        ("raise RuntimeError('fit exploded')", "return X.iloc[:, :1]", "fit exploded"),
        ("pass", "raise RuntimeError('transform exploded')", "transform exploded"),
    ],
)
def test_plugin_execution_failures_are_expected_exit_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fit_body: str,
    transform_body: str,
    expected_error: str,
) -> None:
    (tmp_path / "plugin.py").write_text(
        "class Explodes:\n"
        "    def fit(self, X):\n"
        f"        {fit_body}\n"
        "    def transform(self, X):\n"
        f"        {transform_body}\n",
        encoding="utf-8",
    )
    features = {
        "plugins": [{"name": "boom", "entrypoint": "./plugin.py:Explodes", "params": {}}],
        "sets": [{"name": "set", "source_columns": ["*"], "plugins": ["boom"]}],
    }
    config_path = write_project(tmp_path, features=features)
    prepare(config_path)
    assert features_project(config_path) == 2
    output = capsys.readouterr().out
    assert expected_error in output
    assert "Traceback" not in output


def test_plugin_cannot_mutate_owned_base_frames(tmp_path: Path) -> None:
    (tmp_path / "plugin.py").write_text(
        "import pandas as pd\n"
        "class Mutates:\n"
        "    def fit(self, X): X['income'] = -1\n"
        "    def transform(self, X):\n"
        "        X['income'] = -2\n"
        "        return pd.DataFrame({'derived': X['row_id'] + 1}, index=X.index)\n",
        encoding="utf-8",
    )
    features = {
        "plugins": [{"name": "mutates", "entrypoint": "./plugin.py:Mutates", "params": {}}],
        "sets": [{"name": "safe", "source_columns": ["*"], "plugins": ["mutates"]}],
    }
    config_path = write_project(tmp_path, features=features)
    prepare(config_path)
    assert features_project(config_path) == 0

    for split in ("train", "validation", "test"):
        frame = read_set(tmp_path, "safe", split)
        assert frame["income"].tolist() == [
            1000.0 + row_id * 10 for row_id in frame["row_id"]
        ]


def test_multiple_feature_sets_manifests_lineage_and_global_manifest(
    tmp_path: Path,
) -> None:
    (tmp_path / "plugin.py").write_text(
        "import pandas as pd\n"
        "class Ratio:\n"
        "    def __init__(self, epsilon=0): self.epsilon = epsilon\n"
        "    def fit(self, X): pass\n"
        "    def transform(self, X):\n"
        "        return pd.DataFrame({'income_per_age': X['income'] / (X['age'] + self.epsilon)}, index=X.index)\n",
        encoding="utf-8",
    )
    features = {
        "plugins": [
            {"name": "ratio", "entrypoint": "./plugin.py:Ratio", "params": {"epsilon": 0.1}}
        ],
        "sets": [
            {"name": "base", "source_columns": ["*"], "plugins": []},
            {
                "name": "ratio_set",
                "source_columns": ["age", "income"],
                "plugins": ["ratio"],
            },
        ],
    }
    config_path = write_project(tmp_path, features=features)
    prepare(config_path)

    assert features_project(config_path) == 0
    assert (tmp_path / ".mltool/features/base").is_dir()
    assert (tmp_path / ".mltool/features/ratio_set").is_dir()
    set_manifest = json.loads(
        (tmp_path / ".mltool/features/ratio_set/manifest.json").read_text()
    )
    assert set_manifest["source_columns"] == ["age", "income"]
    assert set_manifest["plugins"][0]["name"] == "ratio"
    assert set_manifest["plugins"][0]["params"] == {"epsilon": 0.1}
    assert set_manifest["plugins"][0]["generated_columns"] == ["income_per_age"]
    assert set_manifest["final_feature_columns"] == ["age", "income", "income_per_age"]
    assert set_manifest["lineage"] == {
        "age": {"source": "base"},
        "income": {"source": "base"},
        "income_per_age": {"source": "plugin", "plugin": "ratio"},
    }
    global_manifest = json.loads(
        (tmp_path / ".mltool/features/manifest.json").read_text()
    )
    assert [item["name"] for item in global_manifest["feature_sets"]] == [
        "base",
        "ratio_set",
    ]
    assert global_manifest["target"] == "label"
    assert global_manifest["rows"] == {"train": 42, "validation": 9, "test": 9}
    assert [item["feature_count"] for item in global_manifest["feature_sets"]] == [
        4,
        3,
    ]
    assert all(
        Path(item["path"]).is_dir() for item in global_manifest["feature_sets"]
    )


def test_missing_prepared_artifacts_requests_prepare(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_project(tmp_path, features=base_features_config())
    assert features_project(config_path) == 2
    output = capsys.readouterr().out
    assert 'run "mltool prepare" first' in output
    assert "Traceback" not in output


def test_modified_source_after_prepare_is_detected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_project(tmp_path, features=base_features_config())
    prepare(config_path)
    source = tmp_path / "data/dataset.csv"
    source.write_bytes(source.read_bytes() + b"\n")

    assert features_project(config_path) == 2
    assert "source dataset changed after preparation" in capsys.readouterr().out


@pytest.mark.parametrize("damage", ["task", "manifest", "missing_split", "schema"])
def test_invalid_prepared_artifacts_fail_clearly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    damage: str,
) -> None:
    config_path = write_project(tmp_path, features=base_features_config())
    prepare(config_path)
    prepared = tmp_path / ".mltool/prepared"
    if damage == "task":
        manifest = json.loads((prepared / "manifest.json").read_text())
        manifest["task"]["target"] = "different"
        (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif damage == "manifest":
        (prepared / "manifest.json").write_text("{broken", encoding="utf-8")
    elif damage == "missing_split":
        (prepared / "validation.parquet").unlink()
    else:
        validation = pd.read_parquet(prepared / "validation.parquet")
        validation["extra"] = 1
        validation.to_parquet(prepared / "validation.parquet", index=False)

    assert features_project(config_path) == 2
    output = capsys.readouterr().out
    assert 'mltool prepare" again' in output or "prepared manifest is unreadable" in output
    assert "Traceback" not in output


def test_failed_rematerialization_preserves_existing_features(
    tmp_path: Path,
) -> None:
    config_path = write_project(tmp_path, features=base_features_config())
    prepare(config_path)
    assert features_project(config_path) == 0
    old_manifest = (tmp_path / ".mltool/features/manifest.json").read_bytes()
    old_base = (tmp_path / ".mltool/features/base/train.parquet").read_bytes()

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["features"] = {
        "plugins": [
            {"name": "missing", "entrypoint": "./missing.py:Missing", "params": {}}
        ],
        "sets": [
            {"name": "new", "source_columns": ["*"], "plugins": ["missing"]}
        ],
    }
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    assert features_project(config_path) == 2
    assert (tmp_path / ".mltool/features/manifest.json").read_bytes() == old_manifest
    assert (tmp_path / ".mltool/features/base/train.parquet").read_bytes() == old_base
    assert not (tmp_path / ".mltool/features/new").exists()


def test_repeated_materialization_is_deterministic(tmp_path: Path) -> None:
    config_path = write_project(tmp_path, features=base_features_config())
    prepare(config_path)
    assert features_project(config_path) == 0
    first_frames = {
        split: read_set(tmp_path, "base", split)
        for split in ("train", "validation", "test")
    }
    first_manifest = (tmp_path / ".mltool/features/manifest.json").read_text()

    assert features_project(config_path) == 0
    for split, first in first_frames.items():
        pd.testing.assert_frame_equal(first, read_set(tmp_path, "base", split))
    assert (tmp_path / ".mltool/features/manifest.json").read_text() == first_manifest


def test_validate_prepare_and_features_cli_boundaries_remain_working(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_project(tmp_path, features=base_features_config())

    assert validate_project(config_path) == 0
    assert "Result: VALID" in capsys.readouterr().out
    assert prepare_project(config_path) == 0
    assert "Result: PREPARED" in capsys.readouterr().out
    assert features_project(config_path) == 0
    assert "Result: MATERIALIZED" in capsys.readouterr().out
