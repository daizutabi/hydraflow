from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from tests.core.conftest import Collect, Results


@pytest.fixture(scope="module")
def results(collect: Collect) -> Results:
    file = Path(__file__).parent / "force_new_run.py"
    return collect(file, ["count=3"], ["count=3"], ["count=3"])


def test_len(results: Results):
    assert len(results) == 3


@pytest.fixture(scope="module", params=range(3))
def result(results: Results, request: pytest.FixtureRequest):
    assert isinstance(request.param, int)
    return results[request.param]


@pytest.fixture(scope="module")
def path(result: tuple[Path, DictConfig]):
    return result[0]


def test_count(path: Path):
    assert path.joinpath("a.txt").read_text() == "3"
