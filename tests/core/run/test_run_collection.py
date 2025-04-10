from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from omegaconf import ListConfig
from polars import DataFrame

from hydraflow.core.run_collection import Run, RunCollection


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
    return RunCollection(runs)


type Rc = RunCollection[Run[Config, Impl]]


def test_repr(rc: Rc):
    assert repr(rc) == "RunCollection(Run[Impl], n=12)"


def test_repr_empty():
    assert repr(RunCollection([])) == "RunCollection(empty)"


def test_len(rc: Rc):
    assert len(rc) == 12


def test_bool(rc: Rc):
    assert bool(rc) is True


def test_getitem_int(rc: Rc):
    assert isinstance(rc[0], Run)


def test_getitem_slice(rc: Rc):
    assert isinstance(rc[:3], RunCollection)


def test_getitem_iterable(rc: Rc):
    assert isinstance(rc[[0, 1, 2]], RunCollection)


def test_iter(rc: Rc):
    assert len(list(iter(rc))) == 12


def test_update(rc: Rc):
    rc.update("size.height", 10)
    assert all(r.get("size.height") is None for r in rc)


def test_update_force(rc: Rc):
    rc.update("size.height", 10, force=True)
    assert all(r.get("size.height") == 10 for r in rc)


def test_update_callable(rc: Rc):
    rc.update("size.height", lambda r: r.get("size.width") + 10, force=True)
    assert all(r.get("size.height") == r.get("size.width") + 10 for r in rc)


def test_filter(rc: Rc):
    assert len(rc.filter(count=1, name="def")) == 3


def test_filter_callable(rc: Rc):
    assert len(rc.filter(lambda r: r.get("count") == 1)) == 6


def test_filter_tuple(rc: Rc):
    assert len(rc.filter(("size.width", 10), ("count", 2))) == 2


def test_filter_tuple_list(rc: Rc):
    assert len(rc.filter(("size.width", [10, 30]))) == 8


def test_filter_tuple_tuple(rc: Rc):
    assert len(rc.filter(("size.width", (20, 30)))) == 8


def test_filter_multi(rc: Rc):
    assert len(rc.filter(("size.width", (20, 30)), count=1, name="abc")) == 2


def test_try_get(rc: Rc):
    assert rc.try_get(("size.height", 10)) is None


def test_try_get_error(rc: Rc):
    with pytest.raises(ValueError):
        rc.try_get(count=1)


def test_get(rc: Rc):
    r = rc.get(("size.width", 10), count=1, name="abc")
    assert r.get("count") == 1
    assert r.get("name") == "abc"
    assert r.get("size.width") == 10


def test_get_error(rc: Rc):
    with pytest.raises(ValueError):
        rc.get(count=100)


def test_first(rc: Rc):
    r = rc.first(count=1, name="abc")
    assert r.get("count") == 1
    assert r.get("name") == "abc"


def test_first_error(rc: Rc):
    with pytest.raises(ValueError):
        rc.first(count=100)


def test_last(rc: Rc):
    r = rc.last(count=2, name="def")
    assert r.get("count") == 2
    assert r.get("name") == "def"


def test_last_error(rc: Rc):
    with pytest.raises(ValueError):
        rc.last(count=100)


def test_to_list(rc: Rc):
    assert sorted(rc.to_list("name")) == [*(["abc"] * 6), *(["def"] * 6)]


def test_to_numpy(rc: Rc):
    assert np.array_equal(rc.to_numpy("count")[3:5], [1, 1])


def test_unique(rc: Rc):
    assert np.array_equal(rc.unique("count"), [1, 2])


def test_n_unique(rc: Rc):
    assert rc.n_unique("size.width") == 3


def test_sort(rc: Rc):
    x = [10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30]
    assert rc.sort("size.width").to_list("size.width") == x
    assert rc.sort("size.width", reverse=True).to_list("size.width") == x[::-1]


def test_sort_emtpy(rc: Rc):
    assert rc.sort().to_list("count")[-1] == 2


def test_sort_multi(rc: Rc):
    r = rc.sort("size.width", "count", reverse=True)[0]
    assert r.get("size.width") == 30
    assert r.get("count") == 2
    assert r.get("name") == "def"


def test_to_frame_default(rc: Rc):
    df = rc.to_frame()
    assert df.shape == (12, 7)


def test_to_frame(rc: Rc):
    df = rc.to_frame("size.width", "count", "run_id")
    assert df.shape == (12, 3)
    assert df.columns == ["size.width", "count", "run_id"]
    assert df.item(0, "size.width") == 10
    assert df.item(0, "count") == 1
    assert df.item(0, "run_id") == "10"
    assert df.item(-1, "size.width") == 30
    assert df.item(-1, "count") == 2
    assert df.item(-1, "run_id") == "30"


def test_to_frame_callable(rc: Rc):
    df = rc.to_frame("count", name=lambda r: r.get("name").upper())
    assert df.item(0, "name") == "ABC"
    assert df.item(-1, "name") == "DEF"


def test_to_frame_callable_struct(rc: Rc):
    df = rc.to_frame("count", x=lambda r: {"a": r.get("name"), "b": r.get("count") + 1})
    assert df.shape == (12, 2)
    df = df.unnest("x")
    assert df.shape == (12, 3)
    assert df.item(0, "a") == "abc"
    assert df.item(-1, "b") == 3


def test_to_frame_callable_list(rc: Rc):
    df = rc.to_frame("count", x=lambda r: [r.get("size.width")] * r.get("count"))
    assert df.shape == (12, 2)
    assert df.item(0, "x").to_list() == [10]
    assert df.item(-1, "x").to_list() == [30, 30]


def test_group_by_dict(rc: Rc):
    gp = rc.group_by("count", "name")
    assert isinstance(gp, dict)
    assert list(gp.keys()) == [(1, "abc"), (1, "def"), (2, "abc"), (2, "def")]
    assert all(len(r) == 3 for r in gp.values())


def test_group_by_frame(rc: Rc):
    df = rc.group_by("count", x=lambda rc: len(rc))
    assert isinstance(df, DataFrame)
    assert df.shape == (2, 2)
    assert df["x"].to_list() == [6, 6]


def test_group_by_frame_multi(rc: Rc):
    df = rc.group_by("count", "name", x=lambda rc: len(rc))
    assert isinstance(df, DataFrame)
    assert df.shape == (4, 3)
    assert df["x"].to_list() == [3, 3, 3, 3]


def test_to_hashable_list_config():
    from hydraflow.core.run_collection import to_hashable

    assert to_hashable(ListConfig([1, 2, 3])) == (1, 2, 3)


def test_to_hashable_ndarray():
    from hydraflow.core.run_collection import to_hashable

    assert to_hashable(np.array([1, 2, 3])) == (1, 2, 3)


def test_to_hashable_fallback_str():
    from hydraflow.core.run_collection import to_hashable

    class C:
        __hash__ = None  # type: ignore

        def __str__(self) -> str:
            return "abc"

        def __iter__(self):
            raise TypeError

    assert to_hashable(C()) == "abc"
