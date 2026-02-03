from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product, starmap
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from polars import DataFrame

from hydraflow.core.run import Run
from hydraflow.core.run_collection import RunCollection

if TYPE_CHECKING:
    from collections.abc import Callable


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


@pytest.fixture(scope="module")
def rc(run_factory: Callable[..., Run[Config, Impl]]):
    it = product([1, 2], ["abc", "def"], [10, 20, 30])
    it = ([Path("/".join(map(str, p))), *p] for p in it)
    runs = list(starmap(run_factory, it))
    return RunCollection[Run[Config, Impl]](runs, Run.get)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]


type Rc = RunCollection[Run[Config, Impl]]


def test_getitem_key(rc: Rc):
    gp = rc.group_by("count")
    assert len(gp[1]) == 6
    assert len(gp[2]) == 6


def test_getitem_key_multi(rc: Rc):
    gp = rc.group_by("count", "name")
    assert len(gp[1, "abc"]) == 3
    assert len(gp[1, "def"]) == 3
    assert len(gp[2, "abc"]) == 3
    assert len(gp[2, "def"]) == 3


def test_iter(rc: Rc):
    gp = rc.group_by("count")
    assert list(gp) == [1, 2]


def test_len(rc: Rc):
    gp = rc.group_by("count")
    assert len(gp) == 2


def test_contains(rc: Rc):
    gp = rc.group_by("count")
    assert 1 in gp
    assert 3 not in gp


def test_keys(rc: Rc):
    gp = rc.group_by("count")
    assert list(gp.keys()) == [1, 2]


def test_values(rc: Rc):
    gp = rc.group_by("count")
    assert len(list(gp.values())) == 2


def test_items(rc: Rc):
    gp = rc.group_by("count")
    assert len(list(gp.items())) == 2


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
    df = rc.group_by("count", "name").agg(x=len)
    assert df.item(0, "x") == 3
    assert df.item(1, "x") == 3
