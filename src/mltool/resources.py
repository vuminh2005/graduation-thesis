"""cgroup v2 resource limits, passed to AutoGluon so it sizes itself correctly.

AutoGluon 1.6 reads memory from ``psutil.virtual_memory()``, i.e. the host
total, so a process capped by ``systemd-run -p MemoryMax=`` or a container
still believes the whole machine is available. Its ``fit(memory_limit=...)``
(GB, a soft limit) and ``fit(num_cpus=...)`` parameters are the supported way to
tell it otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any

from mltool.config import TrainingConfig

CGROUP_ROOT = Path("/sys/fs/cgroup")
GIB = 1024**3


@dataclass(frozen=True)
class CgroupLimits:
    memory_bytes: int | None
    cpus: float | None


@dataclass(frozen=True)
class ResourceLimits:
    """What AutoGluon is told; ``None`` leaves its own "auto" detection in charge."""

    memory_limit_gb: float | None
    num_cpus: int | None
    memory_source: str | None
    cpu_source: str | None
    host_memory_gb: float | None
    host_cpus: int | None
    cgroup_memory_gb: float | None
    cgroup_cpus: float | None

    def fit_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.memory_limit_gb is not None:
            kwargs["memory_limit"] = self.memory_limit_gb
        if self.num_cpus is not None:
            kwargs["num_cpus"] = self.num_cpus
        return kwargs

    def as_record(self) -> dict[str, Any]:
        return {
            "memory_limit_gb": self.memory_limit_gb,
            "memory_source": self.memory_source,
            "num_cpus": self.num_cpus,
            "cpu_source": self.cpu_source,
            "host_memory_gb": self.host_memory_gb,
            "host_cpus": self.host_cpus,
            "cgroup_memory_gb": self.cgroup_memory_gb,
            "cgroup_cpus": self.cgroup_cpus,
        }


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _own_cgroup(proc_cgroup: Path, root: Path) -> Path | None:
    """The unified-hierarchy entry (``0::/path``); absent on cgroup v1."""
    text = _read(proc_cgroup)
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("0::"):
            return root / line[3:].lstrip("/")
    return None


def detect_cgroup_limits(
    root: Path = CGROUP_ROOT, proc_cgroup: Path = Path("/proc/self/cgroup")
) -> CgroupLimits:
    """The tightest finite ``memory.max`` / ``cpu.max`` from this process's cgroup up.

    A limit set on an ancestor (e.g. a user slice) binds as much as one on the
    process's own scope, so every level is read. A missing file means that
    controller is not enabled there, which is the same as no limit.
    """
    leaf = _own_cgroup(proc_cgroup, root)
    if leaf is None:
        return CgroupLimits(memory_bytes=None, cpus=None)
    memory: int | None = None
    cpus: float | None = None
    node = leaf
    while True:
        raw_memory = _read(node / "memory.max")
        if raw_memory and raw_memory != "max":
            try:
                value = int(raw_memory)
            except ValueError:
                value = None
            if value is not None and value > 0:
                memory = value if memory is None else min(memory, value)
        raw_cpu = _read(node / "cpu.max")
        if raw_cpu:
            quota, _, period = raw_cpu.partition(" ")
            if quota != "max":
                try:
                    value_cpus = int(quota) / int(period or 100000)
                except (ValueError, ZeroDivisionError):
                    value_cpus = None
                if value_cpus is not None and value_cpus > 0:
                    cpus = value_cpus if cpus is None else min(cpus, value_cpus)
        if node == root or root not in node.parents:
            break
        node = node.parent
    return CgroupLimits(memory_bytes=memory, cpus=cpus)


def host_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None


def _gb(value: int | None) -> float | None:
    return None if value is None else round(value / GIB, 3)


def resolve_resource_limits(
    training: TrainingConfig,
    *,
    cgroup: CgroupLimits | None = None,
    host_memory: int | None = None,
    host_cpus: int | None = None,
) -> ResourceLimits:
    """Config overrides win; otherwise a finite cgroup limit below the host total."""
    cgroup = cgroup if cgroup is not None else detect_cgroup_limits()
    host_memory = host_memory if host_memory is not None else host_memory_bytes()
    host_cpus = host_cpus if host_cpus is not None else os.cpu_count()

    memory_limit_gb: float | None = None
    memory_source: str | None = None
    if training.memory_limit_gb is not None:
        memory_limit_gb, memory_source = float(training.memory_limit_gb), "config"
    elif cgroup.memory_bytes is not None and (
        host_memory is None or cgroup.memory_bytes < host_memory
    ):
        memory_limit_gb, memory_source = _gb(cgroup.memory_bytes), "cgroup"

    num_cpus: int | None = None
    cpu_source: str | None = None
    if training.num_cpus is not None:
        num_cpus, cpu_source = training.num_cpus, "config"
    elif cgroup.cpus is not None and (host_cpus is None or cgroup.cpus < host_cpus):
        # A fractional quota cannot be expressed as threads; round down so the
        # quota is never oversubscribed, but always allow one.
        num_cpus, cpu_source = max(1, math.floor(cgroup.cpus)), "cgroup"

    return ResourceLimits(
        memory_limit_gb=memory_limit_gb,
        num_cpus=num_cpus,
        memory_source=memory_source,
        cpu_source=cpu_source,
        host_memory_gb=_gb(host_memory),
        host_cpus=host_cpus,
        cgroup_memory_gb=_gb(cgroup.memory_bytes),
        cgroup_cpus=cgroup.cpus,
    )
