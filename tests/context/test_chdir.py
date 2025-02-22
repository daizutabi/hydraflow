import pytest
from mlflow.entities import Run

from hydraflow.entities.run_collection import RunCollection

pytestmark = pytest.mark.xdist_group(name="group4")


@pytest.fixture(scope="module")
def rc(collect):
    return collect("context/chdir.py", ["-m", "count=1,2"])


def test_rc_len(rc: RunCollection):
    assert len(rc) == 2


@pytest.fixture(scope="module", params=[1, 2])
def run(rc: RunCollection, request: pytest.FixtureRequest):
    return rc.get(count=request.param)


def test_run_count(run: Run):
    from hydraflow.core.io import get_artifact_path

    text = get_artifact_path(run, "a.txt").read_text()
    assert text == run.data.params["count"]
