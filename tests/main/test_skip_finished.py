import pytest
from mlflow.entities import Run, RunStatus
from mlflow.tracking import MlflowClient

from hydraflow.entities.run_collection import RunCollection

pytestmark = pytest.mark.xdist_group(name="group4")


@pytest.fixture(scope="module")
def rc(collect):
    client = MlflowClient()
    running = RunStatus.to_string(RunStatus.RUNNING)

    filename = "main/skip_finished.py"
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


@pytest.fixture(scope="module")
def count(run: Run):
    return int(run.data.params["count"])


@pytest.fixture(scope="module")
def text(run: Run):
    from hydraflow.utils import get_artifact_path

    path = get_artifact_path(run, "a.txt")
    return path.read_text()


def test_count(text: str, count: int):
    assert len(text.splitlines()) == count


def test_config(text: str, count: int):
    assert int(text.split(" ", maxsplit=1)[0]) == count


def test_run(text: str, run: Run):
    line = text.splitlines()[-1]
    assert line.split(" ", maxsplit=1)[1] == run.info.run_id
