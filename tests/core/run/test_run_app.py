from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from hydraflow.core.run import Run

if TYPE_CHECKING:
    from hydraflow.core.run_collection import RunCollection
    from tests.conftest import Collect, Results

# pyright: reportUnknownMemberType=false
# pyright: reportAttributeAccessIssue=false


class Db:
    name: str = ""
    b: int = 0


class Config:
    a: int = 0
    db: Db = Db()


@pytest.fixture(scope="module")
def results(collect: Collect):
    file = Path(__file__).parent / "run.py"
    return collect(file, ["count=10", "name=abc", "size.width=1", "size.height=3"])


def test_len(results: Results):
    assert len(results) == 1


def test_config(results: Results):
    run_dir = results[0][0].parent
    run = Run[Config](run_dir)
    assert run.get("count") == 10
    assert run.get("name") == "abc"
    assert run.get("size.width") == 1
    assert run.get("size.height") == 3
    assert run.get("size") == {"width": 1, "height": 3}


class Dummy:
    pass


class ImplConfig:
    path: Path
    cfg: Dummy

    def __init__(self, path: Path, cfg: Dummy):
        self.path = path
        self.cfg = cfg


@pytest.fixture(scope="module")
def run_impl_config(results: Results):
    run_dir: Path = results[0][0].parent
    return Run[Dummy, ImplConfig].load(run_dir, ImplConfig)


def test_impl_config_repr(run_impl_config: Run[Dummy, ImplConfig]):
    assert repr(run_impl_config).startswith("Run[ImplConfig]('")


def test_impl_config(run_impl_config: Run[Dummy, ImplConfig]):
    assert run_impl_config.impl.path.stem == "artifacts"
    cfg = run_impl_config.cfg
    assert cfg.count == 10
    assert cfg.name == "abc"
    assert cfg.size.width == 1
    assert cfg.size.height == 3


def test_chdir(run_impl_config: Run[Dummy, ImplConfig]):
    run = run_impl_config
    with run.chdir():
        Path("a.txt").write_text("a")
    assert run.path("a.txt").read_text() == "a"


@pytest.fixture(scope="module")
def rc(results: Results):
    run_dir: Path = results[0][0].parent
    return Run[Dummy, ImplConfig].load([run_dir, run_dir], ImplConfig)


def test_iterdir_glob(rc: RunCollection[Run[Dummy, ImplConfig]]):
    for run in rc:
        run.path("a.txt").write_text("a")

    assert len(list(rc.iterdir())) == 6
    assert len(list(rc.glob("*.txt"))) == 2
