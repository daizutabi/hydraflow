import pytest
from mlflow.entities import Run

from hydraflow.run_collection import RunCollection
from hydraflow.utils import get_artifact_path

pytestmark = pytest.mark.xdist_group(name="group2")


@pytest.fixture(scope="module")
def rc(collect):
    args = ["-m", "name=a,b,c"]
    return collect("context/context.py", args)


@pytest.fixture(scope="module", params=range(3))
def run(rc: RunCollection, request: pytest.FixtureRequest):
    return rc[request.param]


def test_chdir_artifact(run: Run):
    path = get_artifact_path(run, "b.txt")
    assert path.read_text() == "chdir_artifact"
