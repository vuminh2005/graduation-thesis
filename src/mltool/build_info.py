"""Which MLTool build produced an artifact: the git commit of the imported source."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import subprocess
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None


def source_commit(package_dir: Path) -> dict[str, Any] | None:
    """``{"commit": sha, "dirty": bool}`` for a package directory, or ``None``.

    ``None`` unless the package's own ``__init__.py`` is tracked by the enclosing
    repository: a non-editable install under a project's ``.venv`` sits inside
    that project's git tree without being its code, and must not borrow its HEAD.
    ``dirty`` means uncommitted changes inside the package directory itself, i.e.
    the running code differs from ``commit``. Never raises.
    """
    try:
        if not package_dir.is_dir():
            return None
        tracked = _git(["ls-files", "--error-unmatch", "__init__.py"], package_dir)
        if tracked is None or tracked.returncode != 0:
            return None
        head = _git(["rev-parse", "HEAD"], package_dir)
        status = _git(["status", "--porcelain", "--", "."], package_dir)
        if head is None or status is None or head.returncode != 0 or status.returncode != 0:
            return None
        commit = head.stdout.strip()
        return {"commit": commit, "dirty": bool(status.stdout.strip())} if commit else None
    except Exception:
        return None


@lru_cache(maxsize=1)
def mltool_commit() -> dict[str, Any] | None:
    """The running MLTool's commit, computed once per process."""
    return source_commit(PACKAGE_DIR)
