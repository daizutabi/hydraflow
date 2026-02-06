from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from tests.core.conftest import Collect, Results


@pytest.fixture(scope="module")
def results(collect: Collect):
    file = Path(__file__).parent / "chdir.py"
    return collect(file, ["-m", "count=1,2"])


def test_len(results: Results) -> None:
    assert len(results) == 2


@pytest.fixture(scope="module", params=range(2))
def result(results: Results, request: pytest.FixtureRequest):
    assert isinstance(request.param, int)
    return results[request.param]


def test_first(result: tuple[Path, DictConfig]) -> None:
    path, cfg = result
    assert int(path.joinpath("a.txt").read_text()) == cfg.count
