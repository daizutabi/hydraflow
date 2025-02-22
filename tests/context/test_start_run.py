import pytest
from mlflow.entities import Run

from hydraflow.entities.run_collection import RunCollection
from hydraflow.utils import get_artifact_path

pytestmark = pytest.mark.xdist_group(name="group2")


@pytest.fixture(scope="module")
def rc(collect):
    return collect("context/start_run.py", ["-m", "name=a,b,c"])


def test_rc_len(rc: RunCollection):
    assert len(rc) == 3


@pytest.fixture(scope="module", params=range(3))
def run(rc: RunCollection, request: pytest.FixtureRequest):
    return rc[request.param]


def test_run_first(run: Run):
    path = get_artifact_path(run, "1.txt")
    assert path.read_text() == run.data.params["name"]


def test_run_second(run: Run):
    path = get_artifact_path(run, "2.txt")
    assert path.read_text() == run.data.params["name"] * 2
