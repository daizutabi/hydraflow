from pathlib import Path

import pytest
from omegaconf import DictConfig


@pytest.fixture(scope="module")
def results(collect):
    file = Path(__file__).parent / "chdir.py"
    return collect(file, ["-m", "count=1,2"])


def test_len(results):
    assert len(results) == 2


@pytest.fixture(scope="module", params=range(2))
def result(results, request: pytest.FixtureRequest):
    return results[request.param]


def test_first(result: tuple[Path, DictConfig]):
    path, cfg = result
    assert int(path.joinpath("a.txt").read_text()) == cfg.count
