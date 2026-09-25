"""``mltool score``: predictions for raw rows from one registry version alone.

A registry version holds everything the pipeline fitted: the preprocessor, the
selected FeatureSet's plugins and the predictor, plus the ``scoring`` record
``finalize`` wrote into its manifest (which columns each stage reads, in which
order). Scoring uses only those files: it never reads ``mltool.yaml``,
``.mltool/final`` or the user's source files. The preprocessor and plugins were
saved with cloudpickle, which stored their code by value; code they import from
a sibling module is saved by reference and must still be importable.

It writes nothing but the requested output file, so like ``status`` it is not
recorded in the project's run history (and it works on a registry copy that has
no project around it).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import cloudpickle
import pandas as pd

from mltool.autogluon_adapter import AutoGluonAdapter, AutoGluonError
from mltool.feature_plugins import FeaturePluginError, _transform
from mltool.preprocessing import PreprocessingError, _transform_one

SCORE_SPLIT = "score"  # the split name in contract-violation messages
OUTPUT_FORMATS = {".csv": "csv", ".parquet": "parquet"}


class ScoringError(ValueError):
    """An expected problem with the registry version, the input rows or the output path."""


@dataclass
class ScoreResult:
    version: int
    version_path: Path
    input_path: Path
    output_path: Path
    rows: int
    columns: list[str]
    registered_warning: str | None

    def render(self) -> str:
        lines = [
            "MLTool scoring",
            "",
            f"Registry version {self.version}",
            f"  {self.version_path}",
            "",
            f"Input: {self.input_path} ({self.rows} rows)",
            f"Output: {self.output_path}",
            f"  columns: {', '.join(self.columns)}",
        ]
        if self.registered_warning:
            lines.extend(
                ["", "Warnings", f"  ! this version was registered stale: {self.registered_warning}"]
            )
        lines.extend(["", "Result: SCORED"])
        return "\n".join(lines)


def render_scoring_error(message: str) -> str:
    return "\n".join(["MLTool scoring", "", "Errors", f"  x {message}", "", "Result: FAILED"])


def resolve_version(registry: Path, version: int | None) -> tuple[int, Path]:
    """The requested version's directory, or the latest one."""
    if not registry.is_dir():
        raise ScoringError(f"registry directory not found: {registry}")
    versions = sorted(int(p.name) for p in registry.iterdir() if p.is_dir() and p.name.isdigit())
    if not versions:
        raise ScoringError(f"the registry has no versions: {registry}")
    chosen = versions[-1] if version is None else version
    if chosen not in versions:
        raise ScoringError(
            f"registry version {chosen} does not exist; available: "
            f"{', '.join(str(v) for v in versions)}"
        )
    return chosen, registry / str(chosen)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoringError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScoringError(f"{path} is invalid")
    return value


def load_scoring_record(version: int, version_path: Path) -> dict[str, Any]:
    record = _read_json(version_path / "manifest.json").get("scoring")
    if not isinstance(record, dict):
        raise ScoringError(
            f"registry version {version} predates self-contained scoring (its manifest has no "
            '"scoring" record); run "mltool finalize --force" and "mltool register" to create '
            "a version that can score raw rows"
        )
    return record


def read_rows(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in OUTPUT_FORMATS:
        raise ScoringError(f"input must be a .csv or .parquet file: {path}")
    if not path.is_file():
        raise ScoringError(f"input file not found: {path}")
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as exc:
        raise ScoringError(f"could not read input {path}: {exc}") from exc


def required_columns(record: dict[str, Any]) -> list[str]:
    """What the first stage reads: the preprocessor's fit columns, else the FeatureSet's."""
    if record["preprocessor"]:
        return list(record["raw_feature_columns"])
    feature_set = record["feature_set"]
    return list(dict.fromkeys([*feature_set["source_columns"], *feature_set["plugin_inputs"]]))


def _load_pickle(path: Path, label: str) -> Any:
    try:
        with path.open("rb") as stream:
            return cloudpickle.load(stream)
    except Exception as exc:
        raise ScoringError(f"could not load the fitted {label} {path}: {exc}") from exc


def build_features(record: dict[str, Any], version_path: Path, rows: pd.DataFrame) -> pd.DataFrame:
    """Raw rows -> the predictor's input, replaying finalize's refit transforms in order."""
    needed = required_columns(record)
    missing = [column for column in needed if column not in rows.columns]
    if missing:
        raise ScoringError(
            f"input is missing required column(s): {', '.join(missing)}; this model reads "
            f"{', '.join(needed)}"
        )
    target = record["target"]
    frame = rows.loc[:, needed].copy(deep=True)
    if record["preprocessor"]:
        preprocessor = _load_pickle(version_path / record["preprocessor"], "preprocessor")
        try:
            frame = _transform_one(preprocessor, frame, split_name=SCORE_SPLIT, target=target)
        except PreprocessingError as exc:
            raise ScoringError(str(exc)) from exc

    feature_set = record["feature_set"]
    source_columns = feature_set["source_columns"]
    view_columns = list(dict.fromkeys([*source_columns, *feature_set["plugin_inputs"]]))
    missing = [column for column in view_columns if column not in frame.columns]
    if missing:
        raise ScoringError(
            f"the preprocessor did not produce column(s) the FeatureSet reads: {', '.join(missing)}"
        )
    parts = [frame.loc[:, source_columns]]
    view = frame.loc[:, view_columns]
    for plugin in feature_set["plugins"]:
        fitted = _load_pickle(version_path / plugin["artifact"], f'feature plugin "{plugin["name"]}"')
        try:
            generated = _transform(
                fitted, view, plugin_name=plugin["name"], split_name=SCORE_SPLIT, target=target
            )
        except FeaturePluginError as exc:
            raise ScoringError(str(exc)) from exc
        if generated.columns.tolist() != plugin["generated_columns"]:
            raise ScoringError(
                f'feature plugin "{plugin["name"]}" generated {generated.columns.tolist()!r}, '
                f'not the recorded {plugin["generated_columns"]!r}'
            )
        parts.append(generated)
    features = pd.concat(parts, axis=1)
    if features.columns.tolist() != feature_set["final_feature_columns"]:
        raise ScoringError("the rebuilt feature columns differ from the recorded final feature columns")
    return features


def _write(frame: pd.DataFrame, path: Path) -> None:
    """Atomic: a failed write leaves no partial output file."""
    fd, temporary = tempfile.mkstemp(prefix=".mltool-score-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        if path.suffix.lower() == ".csv":
            frame.to_csv(temporary, index=False)
        else:
            frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except Exception as exc:
        Path(temporary).unlink(missing_ok=True)
        raise ScoringError(f"could not write {path}: {exc}") from exc


def score(
    *,
    registry: Path,
    input_path: Path,
    output_path: Path,
    version: int | None = None,
    adapter_factory: Callable[[], AutoGluonAdapter] | None = None,
) -> ScoreResult:
    if output_path.suffix.lower() not in OUTPUT_FORMATS:
        raise ScoringError(f"output must be a .csv or .parquet file: {output_path}")
    if not output_path.parent.is_dir():
        raise ScoringError(f"output directory does not exist: {output_path.parent}")
    number, version_path = resolve_version(registry, version)
    record = load_scoring_record(number, version_path)
    rows = read_rows(input_path)
    features = build_features(record, version_path, rows)
    try:
        predictions, probabilities = (adapter_factory or AutoGluonAdapter)().predict_saved(
            predictor_path=version_path / record["predictor"],
            features=features,
            task_type=record["task"],
        )
    except AutoGluonError as exc:
        raise ScoringError(f"the registered predictor could not predict: {exc}") from exc

    output = pd.DataFrame(index=rows.index)
    for column in record["id_columns"]:
        if column in rows.columns:
            output[column] = rows[column]
    output["prediction"] = predictions.to_numpy()
    if probabilities is not None:
        for label in probabilities.columns:
            output[f"proba_{label}"] = probabilities[label].to_numpy()
    _write(output, output_path)
    metadata_path = version_path / "metadata.json"
    warning = _read_json(metadata_path).get("warning") if metadata_path.is_file() else None
    return ScoreResult(
        version=number,
        version_path=version_path,
        input_path=input_path,
        output_path=output_path,
        rows=len(output),
        columns=output.columns.tolist(),
        registered_warning=warning,
    )
