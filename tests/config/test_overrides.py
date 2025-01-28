from pathlib import Path

import pytest
from mlflow.artifacts import download_artifacts
from mlflow.entities import Run

from hydraflow.run_collection import RunCollection

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(scope="module")
def rc(collect):
    args = ["-m", "name=a,b", "height=3"]
    return collect("config/overrides.py", args)


@pytest.fixture(scope="module")
def run(rc: RunCollection):
    return rc.first()


def test_select_overrides(run: Run):
    path = download_artifacts(f"{run.info.artifact_uri}/overrides.txt")
    assert Path(path).read_text() == "{'name': 'x', 'height': 2.0}"
