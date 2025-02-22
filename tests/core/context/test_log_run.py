from pathlib import Path

import pytest
from mlflow.entities import Run

from hydraflow.core.io import get_artifact_path
from hydraflow.entities.run_collection import RunCollection

pytestmark = pytest.mark.xdist_group(name="group6")


@pytest.fixture(scope="module")
def rc(collect):
    file = Path(__file__).parent / "log_run.py"
    return collect(file, ["count=100"])


def test_rc_len(rc: RunCollection):
    assert len(rc) == 1


@pytest.fixture(scope="module")
def run(rc: RunCollection):
    return rc.first()


def test_config(run: Run):
    path = get_artifact_path(run, ".hydra/config.yaml")
    assert path.read_text() == "count: 100\n"


def test_overrides(run: Run):
    path = get_artifact_path(run, ".hydra/overrides.yaml")
    assert path.read_text() == "- count=100\n"


@pytest.fixture(scope="module")
def log(run: Run, experiment_name: str):
    path = get_artifact_path(run, f"{experiment_name}.log")
    return path.read_text()


def test_log_info(log: str):
    assert "[__main__][INFO] - log.info" in log


def test_log_exception(log: str):
    assert "[ERROR] - Error during log_run:" in log
    assert "assert cfg.count == 200" in log


def test_log_text(run: Run):
    path = get_artifact_path(run, "text.log")
    assert path.read_text() == "mlflow.log_text\nwrite_text"


def test_log_text_skip_directory(run: Run):
    assert not get_artifact_path(run, "dir.log").exists()
