from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import pytest
from polars import DataFrame

from hydraflow.core.run import Run
from hydraflow.core.run_collection import RunCollection


@dataclass
class Size:
    width: int = 0
    height: int | None = None


@dataclass
class Config:
    count: int = 1
    name: str = "a"
    size: Size = field(default_factory=Size)


class Impl:
    x: int
    y: list[str]

    def __init__(self, path: Path):
        self.x = len(path.as_posix())
        self.y = list(path.parts)


@pytest.fixture(scope="module")
def run_factory():
    def run_factory(path: Path, count: int, name: str, width: int):
        run = Run[Config, Impl](path, Impl)
        run.update("count", count)
        run.update("name", name)
        run.update("size.width", width)
        run.update("size.height", None)
        return run

    return run_factory


@pytest.fixture
def rc(run_factory):
    it = product([1, 2], ["abc", "def"], [10, 20, 30])
    it = ([Path("/".join(map(str, p))), *p] for p in it)
    runs = [run_factory(*p) for p in it]
    return RunCollection(runs, Run.get)


type Rc = RunCollection[Run[Config, Impl], Impl]


def test_getitem_key(rc: Rc):
    gp = rc.group_by("count")
    assert len(gp[1]) == 6
    assert len(gp[2]) == 6


def test_getitem_key_multi(rc: Rc):
    gp = rc.group_by("count", "name")
    assert len(gp[(1, "abc")]) == 3
    assert len(gp[1, "def"]) == 3
    assert len(gp[(2, "abc")]) == 3
    assert len(gp[2, "def"]) == 3


def test_agg_key(rc: Rc):
    df = rc.group_by("count").agg()
    x = DataFrame({"count": [1, 2]})
    x.group_by("count").agg()
    assert df.equals(x)


def test_agg_key_multi(rc: Rc):
    df = rc.group_by("count", "name").agg()
    x = DataFrame({"count": [1, 1, 2, 2], "name": ["abc", "def", "abc", "def"]})
    assert df.equals(x)


def test_agg_get(rc: Rc):
    df = rc.group_by("count", "name").agg("name")
    assert df.item(0, "name").to_list() == ["abc", "abc", "abc"]
    assert df.item(1, "name").to_list() == ["def", "def", "def"]


def test_agg_callable(rc: Rc):
    df = rc.group_by("count", "name").agg(x=lambda x: len(x))
    assert df.item(0, "x") == 3
    assert df.item(1, "x") == 3
