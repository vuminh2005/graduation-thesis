"""Phase-7 SQLite execution state: one row per state-changing mltool invocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

STATUSES = ("SUCCEEDED", "FAILED", "BLOCKED")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    command     TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL CHECK (status IN ('SUCCEEDED', 'FAILED', 'BLOCKED')),
    exit_code   INTEGER,
    details     TEXT NOT NULL DEFAULT '{}'
)
"""


class StateError(RuntimeError):
    """The execution-state database could not be read or written."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def state_path(project_root: Path) -> Path:
    return project_root / ".mltool" / "state.db"


@dataclass
class RunRecord:
    id: int
    command: str
    started_at: str
    finished_at: str | None
    status: str
    exit_code: int | None
    details: dict[str, Any]


@dataclass
class TrackedRun:
    """An in-flight invocation; ``details`` is filled in by the command and stored on finish."""

    project_root: Path
    command: str
    started_at: str = field(default_factory=utc_now)
    details: dict[str, Any] = field(default_factory=dict)

    def finish(self, status: str, exit_code: int | None) -> None:
        # A command that created nothing (e.g. a first-time failed prepare or a
        # validate) must not create ``.mltool/`` just to log itself.
        if not (self.project_root / ".mltool").is_dir():
            return
        try:
            record_run(
                self.project_root,
                command=self.command,
                started_at=self.started_at,
                finished_at=utc_now(),
                status=status,
                exit_code=exit_code,
                details=self.details,
            )
        except (StateError, OSError) as exc:  # State is best-effort, never a gate.
            print(f"MLTool state warning: could not record run: {exc}", file=sys.stderr)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=10)
    connection.execute(_SCHEMA)
    return connection


def record_run(
    project_root: Path,
    *,
    command: str,
    started_at: str,
    finished_at: str | None,
    status: str,
    exit_code: int | None,
    details: dict[str, Any],
) -> int:
    if status not in STATUSES:
        raise StateError(f"unknown run status {status!r}")
    path = state_path(project_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = _connect(path)
        try:
            with connection:
                cursor = connection.execute(
                    "INSERT INTO runs (command, started_at, finished_at, status, exit_code, details)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        command,
                        started_at,
                        finished_at,
                        status,
                        exit_code,
                        json.dumps(details, sort_keys=True, default=str),
                    ),
                )
                return int(cursor.lastrowid)
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise StateError(str(exc)) from exc


def _read(project_root: Path, query: str, params: tuple[Any, ...] = ()) -> list[RunRecord]:
    """Read-only access: never creates the database."""
    path = state_path(project_root)
    if not path.is_file():
        return []
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
        try:
            rows = connection.execute(query, params).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise StateError(f"could not read {path}: {exc}") from exc
    return [
        RunRecord(
            id=row[0],
            command=row[1],
            started_at=row[2],
            finished_at=row[3],
            status=row[4],
            exit_code=row[5],
            details=json.loads(row[6] or "{}"),
        )
        for row in rows
    ]


_COLUMNS = "id, command, started_at, finished_at, status, exit_code, details"


def list_runs(project_root: Path, limit: int | None = None) -> list[RunRecord]:
    query = f"SELECT {_COLUMNS} FROM runs ORDER BY id DESC"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return _read(project_root, query)


def last_run_per_command(project_root: Path) -> dict[str, RunRecord]:
    rows = _read(
        project_root,
        f"SELECT {_COLUMNS} FROM runs WHERE id IN (SELECT MAX(id) FROM runs GROUP BY command)",
    )
    return {row.command: row for row in rows}
