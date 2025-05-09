from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
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


def test_preload(rc: Rc):
    assert rc.preload(cfg=True, impl=True) is rc


def test_preload_n_jobs(rc: Rc):
    assert rc.preload(cfg=True, impl=True, n_jobs=2) is rc


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


def test_filter_underscore(rc: Rc):
    assert len(rc.filter(size__width=10, count=2)) == 2


def test_filter_tuple_list(rc: Rc):
    assert len(rc.filter(("size.width", [10, 30]))) == 8


def test_filter_underscope_list(rc: Rc):
    assert len(rc.filter(size__width=[10, 30])) == 8


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


def test_to_list_default(rc: Rc):
    assert sorted(rc.to_list("unknown", 1)) == [1] * 12


def test_to_list_default_callable(rc: Rc):
    x = sorted(rc.to_list("unknown", lambda r: r.get("count")))
    assert x == [1] * 6 + [2] * 6


def test_to_numpy(rc: Rc):
    assert np.array_equal(rc.to_numpy("count")[3:5], [1, 1])


def test_to_series(rc: Rc):
    s = rc.to_series("count")
    assert s.to_list() == [1] * 6 + [2] * 6
    assert s.name == "count"


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


def test_map(rc: Rc):
    def func(r: Run[Config, Impl], x: int, y: int = 2) -> int:
        return r.cfg.size.width + x + y

    x = list(rc.map(func, 10, 20))
    assert x == [40, 50, 60] * 4


@pytest.mark.parametrize("progress", [False, True])
def test_pmap(rc: Rc, progress: bool):
    def func(r: Run[Config, Impl], x: int, y: int = 2) -> int:
        return r.cfg.size.width + x + y

    x = rc.pmap(func, x=10, y=20, n_jobs=2, backend="threading", progress=progress)
    assert x == [40, 50, 60] * 4


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


def test_to_frame_kwargs(rc: Rc):
    def func(r: Run[Config, Impl]) -> int:
        return r.cfg.count

    df = rc.to_frame("count", func=func)
    assert df.shape == (12, 2)
    assert df["count"].to_list() == df["func"].to_list()


def test_group_by(rc: Rc):
    from hydraflow.core.group_by import GroupBy

    gp = rc.group_by("count", "name")
    assert isinstance(gp, GroupBy)


def test_concat(rc: Rc):
    def func(r: Run[Config, Impl]) -> DataFrame:
        return DataFrame({"a": [r.get("count"), 20]})

    df = rc.concat(func, "size.width", ("z", lambda r: r.get("count") * 20))
    assert df.shape == (24, 3)
    assert df["a"].to_list()[:4] == [1, 20, 1, 20]
    assert df["size.width"].to_list()[-6:] == [10, 10, 20, 20, 30, 30]
    assert df["z"].to_list()[:6] == [20, 20, 20, 20, 20, 20]
    assert df["z"].to_list()[-6:] == [40, 40, 40, 40, 40, 40]


def test_impls(rc: Rc):
    impls = rc.impls
    assert len(impls) == 12
    assert len(impls.filter(lambda i: i.y[0] == "1")) == 6
