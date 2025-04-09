from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig
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
    df = rc.group_by("count", "name", x=lambda rc: len(rc))
    assert isinstance(df, DataFrame)
    assert df["x"].to_list() == [3, 3, 3, 3]


# # @pytest.fixture(scope="module")
# # def rc(results: list[tuple[Path, DictConfig]]):
# #     return results.filter(lambda r: r.cfg.name == "abc")


# # class Db:
# #     name: str
# #     b: int


# # class Config:
# #     a: int
# #     db: Db


# # @pytest.fixture
# # def rc(root_dirs: list[str]):
# #     return RunConfig[Config].load(root_dirs)


# # def test_run_config_len(rc: RunCollection[RunConfig[Config]]):
# #     assert len(rc) == 5


# # def test_run_config_cfg(rc: RunCollection[RunConfig[Config]]):
# #     assert all(not r.cfg for r in rc)


# # def test_group_by_list_config(rc: RunCollection[RunConfig[Config]]):
# #     rc.set_default("a", lambda x: list(x.script.grid_size))
# #     gp = rc.group_by("a")
# #     assert list(gp.keys()) == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]


# # def test_group_by_list(rc: RunCollection[RunConfig[Config]]):
# #     for r in rc:
# #         r.dummy = [1, 2]  # type: ignore
# #     gp = rc.group_by("dummy")
# #     assert list(gp.keys()) == [(1, 2)]


# # def test_group_by_numpy(rc: RunCollection[RunConfig[Config]]):
# #     for r in rc:
# #         r.dummy = np.array([1, 2])  # type: ignore
# #     gp = rc.group_by("dummy")
# #     assert list(gp.keys()) == [(1, 2)]


# # def test_run_config_set_defaults(rc: RunCollection[RunConfig[Config]]):
# #     rc.set_default("db.name", "abc")
# #     assert all(r.cfg.db.name == "abc" for r in rc)
# #     rc.set_default("a", 10)
# #     assert all(r.cfg.a == 10 for r in rc)


# # def test_run_config_set_defaults_callable(rc: RunCollection[RunConfig[Config]]):
# #     rc.set_default("a", lambda x: x.script.grid_size[0])
# #     assert rc.to_list("a") == [1, 2, 3, 1, 1]
# #     rc.set_default("db.b", lambda x: x.script.grid_size[1])
# #     assert rc.to_list("db.b") == [2, 3, 4, 2, 2]


# # def test_run_config_set_defaults_tuple_callable(rc: RunCollection[RunConfig[Config]]):
# #     rc.set_default(("a", "db.b"), lambda x: x.script.grid_size[:2])
# #     assert rc.to_list("a") == [1, 2, 3, 1, 1]
# #     assert rc.to_list("db.b") == [2, 3, 4, 2, 2]
# #     rc.set_default(("a", "db.b"), lambda x: x.script.cell_size[:2])
# #     assert rc.to_list("a") == [1, 2, 3, 1, 1]
# #     assert rc.to_list("db.b") == [2, 3, 4, 2, 2]
# #     for r in rc[:2]:
# #         r.cfg.a = None  # type: ignore
# #     for r in rc[3:]:
# #         r.cfg.db.b = None  # type: ignore
# #     rc.set_default(("a", "db.b"), lambda x: x.script.grid_size[1:])
# #     assert rc.to_list("a") == [2, 3, 3, 1, 1]
# #     assert rc.to_list("db.b") == [2, 3, 4, 3, 3]


# # def test_run_series_set_defaults_type_error(root_dirs: list[str]):
# #     rs = RunSeries.load(root_dirs)
# #     with pytest.raises(AttributeError):
# #         rs.set_default("a", 3)  # type: ignore


# def test_to_hashable_fallback_str():
#     from hydraflow.core.run_collection import to_hashable

#     class C:
#         __hash__ = None  # type: ignore

#         def __str__(self) -> str:
#             return "abc"

#         def __iter__(self):
#             raise TypeError

#     assert to_hashable(C()) == "abc"
