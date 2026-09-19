"""Structured validation result and terminal rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def display_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value)


@dataclass
class DatasetSummary:
    path: Path
    format: str | None = None
    rows: int | None = None
    columns: int | None = None
    fingerprint: str | None = None
    schema: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class ValidationReport:
    project_name: str | None = None
    task_type: str | None = None
    target: str | None = None
    dataset: DatasetSummary | None = None
    target_distribution: list[tuple[Any, int]] = field(default_factory=list)
    target_kind: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = ["MLTool validation"]

        if self.project_name is not None:
            lines.extend(
                [
                    "",
                    "Project",
                    f"  Name: {self.project_name}",
                    f"  Task: {self.task_type}",
                    f"  Target: {self.target}",
                ]
            )

        if self.dataset is not None:
            data = self.dataset
            lines.extend(["", "Dataset", f"  Path: {data.path}"])
            if data.format is not None:
                lines.append(f"  Format: {data.format}")
            if data.rows is not None:
                lines.append(f"  Rows: {data.rows}")
            if data.columns is not None:
                lines.append(f"  Columns: {data.columns}")
            if data.fingerprint is not None:
                lines.append(f"  SHA256: {data.fingerprint}")
            if data.schema:
                lines.extend(["", "Schema"])
                lines.extend(f"  {name}: {dtype}" for name, dtype in data.schema)

        if self.target_distribution:
            lines.extend(["", "Target"])
            if self.target_kind is not None:
                lines.append(f"  Classes: {self.target_kind}")
            lines.append("  Distribution:")
            lines.extend(
                f"    {display_value(value)}: {count}"
                for value, count in self.target_distribution
            )

        if self.warnings:
            lines.extend(["", "Warnings"])
            lines.extend(f"  ! {warning}" for warning in self.warnings)

        if self.errors:
            lines.extend(["", "Errors"])
            lines.extend(f"  x {error}" for error in self.errors)

        lines.extend(["", f'Result: {"VALID" if self.is_valid else "INVALID"}'])
        return "\n".join(lines)

