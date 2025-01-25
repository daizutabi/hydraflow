from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from mlflow.artifacts import download_artifacts
from mlflow.entities import Run

from hydraflow.run_collection import RunCollection

if TYPE_CHECKING:
    from .utils import Config


@pytest.fixture(scope="module")
def rc(collect):
    args = ["-m", "name=a,b", "age=10"]
    return collect("utils/utils.py", args)


@pytest.fixture(scope="module")
def run(rc: RunCollection):
    return rc.first()


def test_hydra_output_dir(run: Run):
    from hydraflow.utils import get_hydra_output_dir

    path = download_artifacts(f"{run.info.artifact_uri}/hydra_output_dir.txt")
    assert get_hydra_output_dir(run).as_posix() == Path(path).read_text()


def test_load_config(run: Run):
    from hydraflow.utils import load_config

    cfg: Config = load_config(run)  # type: ignore
    assert cfg.name == "a"
    assert cfg.age == 10
    assert cfg.height == 1.7


def test_get_overrides(run: Run):
    path = download_artifacts(f"{run.info.artifact_uri}/overrides.txt")
    assert Path(path).read_text() == "['name=a', 'age=10']"


def test_load_overrides(run: Run):
    from hydraflow.utils import load_overrides

    overrides = load_overrides(run)
    assert overrides == ["name=a", "age=10"]
