from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from hydraflow.core.run import Run

if TYPE_CHECKING:
    from hydraflow.core.run_collection import RunCollection
    from tests.core.conftest import Collect, Results

# pyright: reportUnknownMemberType=false
# pyright: reportAttributeAccessIssue=false


class Db:
    name: str = ""
    b: int = 0


class Config:
    a: int = 0
    db: Db = Db()


@pytest.fixture(scope="module", params=["file", "sqlite"])
def results(collect: Collect, request: pytest.FixtureRequest) -> Results:
    file = Path(__file__).parent / f"run_{request.param}.py"
    return collect(file, ["count=10", "name=abc", "size.width=1", "size.height=3"])


def test_len(results: Results) -> None:
    assert len(results) == 1


def test_config(results: Results) -> None:
    run_dir = results[0][0].parent
    run = Run[Config](run_dir)
    assert run.get("count") == 10
    assert run.get("name") == "abc"
    assert run.get("size.width") == 1
    assert run.get("size.height") == 3
    assert run.get("size") == {"width": 1, "height": 3}


class Dummy:
    pass


@dataclass
class ImplConfig:
    path: Path
    cfg: Dummy


@pytest.fixture(scope="module")
def run_impl_config(results: Results):
    run_dir: Path = results[0][0].parent
    return Run[Dummy, ImplConfig].load(run_dir, ImplConfig)


def test_impl_config_repr(run_impl_config: Run[Dummy, ImplConfig]) -> None:
    assert repr(run_impl_config).startswith("Run[ImplConfig]('")


def test_impl_config(run_impl_config: Run[Dummy, ImplConfig]) -> None:
    assert run_impl_config.impl.path.stem == "artifacts"
    cfg = run_impl_config.cfg
    assert cfg.count == 10
    assert cfg.name == "abc"
    assert cfg.size.width == 1
    assert cfg.size.height == 3


def test_chdir(run_impl_config: Run[Dummy, ImplConfig]) -> None:
    run = run_impl_config
    with run.chdir():
        Path("a.txt").write_text("a", encoding="utf-8")
    assert run.path("a.txt").read_text() == "a"


@pytest.fixture(scope="module")
def rc(results: Results):
    run_dir: Path = results[0][0].parent
    return Run[Dummy, ImplConfig].load([run_dir, run_dir], ImplConfig)


def test_iterdir_glob(rc: RunCollection[Run[Dummy, ImplConfig]]) -> None:
    for run in rc:
        run.path("a.txt").write_text("a")

    assert len(list(rc.iterdir())) == 6
    assert len(list(rc.glob("*.txt"))) == 2
