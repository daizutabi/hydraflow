from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from tests.core.conftest import Collect, Results


@pytest.fixture(scope="module")
def results(collect: Collect):
    file = Path(__file__).parent / "start_run.py"
    return collect(file, ["-m", "name=a,b,c"])


def test_len(results: Results):
    assert len(results) == 3


@pytest.fixture(scope="module", params=range(3))
def result(results: Results, request: pytest.FixtureRequest):
    assert isinstance(request.param, int)
    return results[request.param]


def test_first(result: tuple[Path, DictConfig]):
    path, cfg = result
    assert path.joinpath("1.txt").read_text() == cfg.name


def test_second(result: tuple[Path, DictConfig]):
    path, cfg = result
    assert path.joinpath("2.txt").read_text() == cfg.name * 2
