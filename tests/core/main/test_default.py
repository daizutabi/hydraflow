from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from omegaconf import DictConfig

from hydraflow.core.run import Run

if TYPE_CHECKING:
    from tests.conftest import Collect, Results


@pytest.fixture(scope="module")
def results(collect: Collect) -> Results:
    file = Path(__file__).parent / "default.py"
    return collect(file, ["-m", "count=1,2"], ["-m", "name=a", "count=1,2,3,4"])


def test_len(results: Results):
    assert len(results) == 4


@pytest.fixture(scope="module", params=range(4))
def result(results: Results, request: pytest.FixtureRequest):
    assert isinstance(request.param, int)
    return results[request.param]


@pytest.fixture(scope="module")
def path(result: tuple[Path, DictConfig]):
    return result[0]


@pytest.fixture(scope="module")
def cfg(result: tuple[Path, DictConfig]):
    return result[1]


@pytest.fixture(scope="module")
def count(cfg: DictConfig):
    return int(cfg.count)


@pytest.fixture(scope="module")
def text(path: Path):
    return path.joinpath("a.txt").read_text()


def test_count(text: str, count: int):
    assert text == str(count)


@pytest.fixture(scope="module")
def cwd(path: Path):
    return Path(path.joinpath("b.txt").read_text())


def test_cwd(cwd: Path, path: Path):
    assert cwd.name == path.parents[3].name


def test_run(path: Path, cfg: DictConfig):
    run = Run[DictConfig](path.parent)
    assert run.cfg == cfg
