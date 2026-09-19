"""Tabular dataset loading and file-level metadata."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from mltool.config import DataConfig


class DataLoadError(ValueError):
    """An expected failure while locating or reading a configured dataset."""


@dataclass
class LoadedDataset:
    path: Path
    format: str
    frame: pd.DataFrame
    column_names: list[str]
    dtypes: list[str]
    fingerprint: str

    @property
    def row_count(self) -> int:
        return len(self.frame.index)

    @property
    def column_count(self) -> int:
        return len(self.column_names)


def _detect_format(config: DataConfig) -> str:
    if config.format != "auto":
        return config.format
    suffix = config.path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".parquet":
        return "parquet"
    raise DataLoadError(
        f'cannot infer data format from extension "{config.path.suffix or "<none>"}"; '
        "use a .csv or .parquet file, or set data.format explicitly"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DataLoadError(f"could not read dataset file {path}: {exc}") from exc
    return digest.hexdigest()


def _csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            # Match pandas.read_csv's default comma delimiter. Auto-sniffing a
            # different delimiter here would make raw header metadata disagree
            # with the DataFrame parsed below. pandas also skips blank lines.
            for row in csv.reader(stream, dialect=csv.excel):
                if row:
                    return row
            return []
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DataLoadError(f"could not read CSV header from {path}: {exc}") from exc


def _parquet_header(path: Path) -> list[str]:
    try:
        return list(pq.read_schema(path).names)
    except Exception as exc:
        raise DataLoadError(f"could not read Parquet dataset {path}: {exc}") from exc


def _duplicates(names: list[str]) -> list[str]:
    return list(dict.fromkeys(name for name in names if names.count(name) > 1))


def load_dataset(config: DataConfig) -> LoadedDataset:
    path = config.path
    if not path.exists():
        raise DataLoadError(f"dataset path does not exist: {path}")
    if not path.is_file():
        raise DataLoadError(f"dataset path is not a file: {path}")

    data_format = _detect_format(config)
    fingerprint = _sha256(path)

    try:
        if data_format == "csv":
            column_names = _csv_header(path)
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                frame = pd.DataFrame()
        elif data_format == "parquet":
            column_names = _parquet_header(path)
            duplicate_names = _duplicates(column_names)
            if duplicate_names:
                formatted = ", ".join(f'"{name}"' for name in duplicate_names)
                raise DataLoadError(f"duplicate column names found: {formatted}")
            frame = pd.read_parquet(path, engine="pyarrow")
        else:  # Defensive: config validation normally prevents this path.
            raise DataLoadError(f'unsupported data format "{data_format}"')
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"could not read {data_format.upper()} dataset {path}: {exc}") from exc

    # pandas mangles duplicate CSV headers, so keep the independently read names
    # for validation while using the frame's dtypes where names are unambiguous.
    if len(column_names) == len(frame.columns) and len(set(column_names)) == len(column_names):
        frame.columns = column_names
    # Keep dtypes positional so duplicate raw names still retain their actual,
    # independently parsed types in an invalid-dataset report.
    dtypes = [str(dtype) for dtype in frame.dtypes]

    return LoadedDataset(
        path=path,
        format=data_format,
        frame=frame,
        column_names=column_names,
        dtypes=dtypes,
        fingerprint=fingerprint,
    )
