from pathlib import Path

import pytest
from omegaconf import DictConfig


@pytest.fixture(scope="module")
def results(collect):
    file = Path(__file__).parent / "start_run.py"
    return collect(file, ["-m", "name=a,b,c"])


def test_len(results):
    assert len(results) == 3


@pytest.fixture(scope="module", params=range(3))
def result(results, request: pytest.FixtureRequest):
    return results[request.param]


def test_first(result: tuple[Path, DictConfig]):
    path, cfg = result
    assert path.joinpath("1.txt").read_text() == cfg.name


def test_second(result: tuple[Path, DictConfig]):
    path, cfg = result
    assert path.joinpath("2.txt").read_text() == cfg.name * 2
