import pytest
from mlflow.entities import Run, RunStatus
from mlflow.tracking import MlflowClient

from hydraflow.run_collection import RunCollection

pytestmark = pytest.mark.xdist_group(name="group4")


@pytest.fixture(scope="module")
def rc(collect):
    client = MlflowClient()
    running = RunStatus.to_string(RunStatus.RUNNING)

    filename = "context/preemption.py"
    args = ["-m", "count=1,2,3"]

    rc = collect(filename, args)
    client.set_terminated(rc.get(count=2).info.run_id, status=running)
    client.set_terminated(rc.get(count=3).info.run_id, status=running)
    rc = collect(filename, args)
    client.set_terminated(rc.get(count=3).info.run_id, status=running)
    return collect(filename, args)


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
