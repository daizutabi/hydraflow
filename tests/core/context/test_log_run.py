from pathlib import Path

import pytest

pytestmark = pytest.mark.xdist_group(name="group6")


@pytest.fixture(scope="module")
def results(collect):
    file = Path(__file__).parent / "log_run.py"
    return collect(file, ["count=100"])


def test_len(results):
    assert len(results) == 1


@pytest.fixture(scope="module")
def result(results):
    return results[0]


@pytest.fixture(scope="module")
def path(result):
    return result[0]


@pytest.fixture(scope="module")
def log(path: Path, experiment_name: str):
    return path.joinpath(f"{experiment_name}.log").read_text()


def test_log_info(log: str):
    assert "[__main__][INFO] - log.info" in log


def test_log_exception(log: str):
    assert "[ERROR] - Error during log_run:" in log
    assert "assert cfg.count == 200" in log


def test_log_text(path: Path):
    assert path.joinpath("text.log").read_text() == "mlflow.log_text\nwrite_text"


def test_log_text_skip_directory(path: Path):
    assert not path.joinpath("dir.log").exists()
