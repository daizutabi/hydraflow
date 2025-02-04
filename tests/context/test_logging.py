import pytest
from mlflow.entities import Run

from hydraflow.run_collection import RunCollection
from hydraflow.utils import get_artifact_path

pytestmark = pytest.mark.xdist_group(name="group6")


@pytest.fixture(scope="module")
def rc(collect):
    collect("context/logging.py", ["count=100"])
    return collect("context/logging.py", ["count=100"])


def test_rc_len(rc: RunCollection):
    assert len(rc) == 1


@pytest.fixture(scope="module")
def run(rc: RunCollection):
    return rc[0]


@pytest.fixture(scope="module")
def hydra_log(run: Run, experiment_name: str):
    path = get_artifact_path(run, f"{experiment_name}.log")
    return path.read_text()


@pytest.mark.parametrize(
    ("i", "suffix"),
    [(0, "] - first"), (1, "] - 100"), (2, "] - second"), (3, "] - 100")],
)
def test_hydra_log(hydra_log: str, i: int, suffix: str):
    assert hydra_log.splitlines()[i].endswith(suffix)


def test_text_log(run: Run):
    path = get_artifact_path(run, "text.log")
    assert path.read_text() == "text\ntext\n"


def test_dir_log(run: Run):
    assert not get_artifact_path(run, "dir.log").exists()


def test_config(run: Run):
    path = get_artifact_path(run, ".hydra/config.yaml")
    cfg = path.read_text()
    assert cfg == "count: 100\n"
