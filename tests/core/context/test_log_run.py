from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from tests.core.conftest import Collect, Results


@pytest.fixture(scope="module")
def results(collect: Collect):
    file = Path(__file__).parent / "log_run.py"
    return collect(file, ["count=100"])


def test_len(results: Results) -> None:
    assert len(results) == 1


@pytest.fixture(scope="module")
def result(results: Results):
    return results[0]


@pytest.fixture(scope="module")
def path(result: tuple[Path, DictConfig]):
    return result[0]


@pytest.fixture(scope="module")
def log(path: Path):
    experiment_name = path.parts[-5]
    return path.joinpath(f"{experiment_name}.log").read_text()


def test_log_info(log: str) -> None:
    assert "[__main__][INFO] - logger.info" in log


def test_log_exception(log: str) -> None:
    assert "[ERROR] - Error during log_run:" in log
    assert "assert cfg.count == 200" in log


def test_log_text(path: Path) -> None:
    assert path.joinpath("text.log").read_text() == "mlflow.log_text\nwrite_text"


def test_log_text_skip_directory(path: Path) -> None:
    assert not path.joinpath("dir.log").exists()
