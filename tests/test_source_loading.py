"""User files run from their current bytes, never from a stale ``__pycache__``.

The import system validates a ``.pyc`` only by the source's size and mtime in
whole seconds, so an edit that keeps the size within the same second used to run
the old code while ``source_sha256`` already hashed the new file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from mltool.config import ExternalPreprocessingConfig, FeaturePluginConfig, source_sha256
from mltool.custom_models import load_entrypoint
from mltool.feature_plugins import load_feature_plugin
from mltool.preprocessing import load_external_preprocessor

SOURCE = '''
import pandas as pd


class Pre:
    def fit(self, X):
        return self

    def transform(self, X):
        return X * 2.0


class Plug:
    def fit(self, X):
        return self

    def transform(self, X):
        return pd.DataFrame({"y": X["x"] * 2.0}, index=X.index)


def factor():
    return 2.0
'''


def same_size_same_second_edit(path: Path) -> None:
    before = path.stat()
    path.write_text(path.read_text().replace("2.0", "3.0"))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size and path.stat().st_mtime_ns == before.st_mtime_ns


def test_each_loader_runs_the_edited_code_and_writes_no_pycache(tmp_path: Path) -> None:
    source = tmp_path / "user.py"
    source.write_text(SOURCE)
    config_path = tmp_path / "mltool.yaml"
    frame = pd.DataFrame({"x": [1.0, 2.0]})

    def outputs() -> tuple[float, float, float]:
        pre, _, _ = load_external_preprocessor(
            ExternalPreprocessingConfig(enabled=True, entrypoint="./user.py:Pre"), config_path
        )
        plug, _ = load_feature_plugin(FeaturePluginConfig("plug", "./user.py:Plug"), config_path)
        return (float(pre.transform(frame)["x"].iloc[0]), float(plug.transform(frame)["y"].iloc[0]),
                load_entrypoint(f"{source}:factor")())

    hash_before = source_sha256(source)
    assert outputs() == (2.0, 2.0, 2.0)
    same_size_same_second_edit(source)
    assert source_sha256(source) != hash_before
    assert outputs() == (3.0, 3.0, 3.0)  # the code that runs is the code that is hashed
    assert not (tmp_path / "__pycache__").exists()
