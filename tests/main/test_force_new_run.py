import pytest
from mlflow.entities import Run

from hydraflow.run_collection import RunCollection

pytestmark = pytest.mark.xdist_group(name="group6")


@pytest.fixture(scope="module")
def rc(collect):
    for _ in range(3):
        rc = collect("main/force_new_run.py", ["count=3"])
    return rc


def test_rc_len(rc: RunCollection):
    assert len(rc) == 3


def test_rc_filter(rc: RunCollection):
    assert len(rc.filter(count=3)) == 3


@pytest.fixture(scope="module", params=range(3))
def run(rc: RunCollection, request: pytest.FixtureRequest):
    return rc[request.param]


def test_count(run: Run):
    from hydraflow.utils import get_artifact_path

    path = get_artifact_path(run, "a.txt")
    assert path.read_text() == "3"
