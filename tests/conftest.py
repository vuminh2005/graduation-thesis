import pytest

import mltool.resources as resources


@pytest.fixture(autouse=True)
def no_cgroup_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite host-independent: a CI box under a cgroup limit must not
    change the fit kwargs that tests assert on. Tests of the detection itself
    call it with an explicit fake cgroup root instead."""
    monkeypatch.setattr(
        resources, "detect_cgroup_limits",
        lambda *args, **kwargs: resources.CgroupLimits(memory_bytes=None, cpus=None),
    )
