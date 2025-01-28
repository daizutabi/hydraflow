import pytest
from mlflow.entities import Run

from hydraflow.run_collection import RunCollection
from hydraflow.utils import get_artifact_path, get_hydra_output_dir


@pytest.fixture(scope="module")
def rc(collect):
    args = ["-m", "name=a,b,c"]
    return collect("context/context.py", args)


@pytest.fixture(scope="module", params=range(3))
def run(rc: RunCollection, request: pytest.FixtureRequest):
    return rc[request.param]


def test_chdir_hydra_output(run: Run):
    path = get_hydra_output_dir(run)
    assert (path / "a.txt").read_text() == "chdir_hydra_output"


def test_chdir_artifact(run: Run):
    path = get_artifact_path(run, "b.txt")
    assert path.read_text() == "chdir_artifact"


def test_log_run(run: Run):
    path = get_artifact_path(run, "a.txt")
    assert path.read_text() == "chdir_hydra_output"
