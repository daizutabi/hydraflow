from pathlib import Path

import pytest

from hydraflow.entities.run_collection import RunCollection

pytestmark = pytest.mark.xdist_group(name="group7")


@pytest.fixture(scope="module")
def rc(collect):
    file = Path(__file__).parent / "rerun_finished.py"
    for _ in range(3):
        rc = collect(file, ["count=3"])
    return rc


def test_rc_len(rc: RunCollection):
    assert len(rc) == 1


def test_count(rc: RunCollection):
    from hydraflow.core.io import get_artifact_path

    run = rc.get(count=3)
    path = get_artifact_path(run, "a.txt")
    assert path.read_text() == "333"
