from pathlib import Path

import pytest
from mlflow.entities import Run

from hydraflow.entities.run_collection import RunCollection

pytestmark = pytest.mark.xdist_group(name="group5")


@pytest.fixture(scope="module")
def rc(collect):
    collect("main/default.py", ["-m", "count=1,2"])
    return collect("main/default.py", ["-m", "name=a", "count=1,2,3,4"])


def test_rc_len(rc: RunCollection):
    assert len(rc) == 4


@pytest.fixture(scope="module", params=[1, 2, 3, 4])
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


def test_run_id(run: Run, text: str):
    assert text.split(",")[0] == run.info.run_id


def test_count(text: str, count: int):
    assert text.split(",")[1] == str(count)


@pytest.fixture(scope="module")
def cwd(run: Run):
    from hydraflow.utils import get_artifact_path

    path = get_artifact_path(run, "b.txt")
    return Path(path.read_text())


def test_cwd(cwd: Path, experiment_name: str):
    assert cwd.name == experiment_name


def test_equals_invalid():
    from hydraflow.core.main import equals

    assert equals(Path("test"), {"a": 1}, None) is False
