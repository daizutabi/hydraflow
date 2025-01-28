import pytest
from mlflow.entities import Run

from hydraflow.run_collection import RunCollection

pytestmark = pytest.mark.xdist_group(name="group3")


@pytest.fixture(scope="module")
def rc(collect):
    collect("context/rerun.py", ["-m", "count=1,2,3"])
    collect("context/rerun.py", ["-m", "count=2,3"])
    return collect("context/rerun.py", ["-m", "count=3"])


def test_rc_len(rc: RunCollection):
    assert len(rc) == 3


@pytest.fixture(scope="module", params=[1, 2, 3])
def run(rc: RunCollection, request: pytest.FixtureRequest):
    return rc.get(count=request.param)


def test_run_count(run: Run):
    from hydraflow.utils import get_artifact_path

    count = int(run.data.params["count"])
    path = get_artifact_path(run, "a.txt")
    text = path.read_text()
    assert len(text) == count
