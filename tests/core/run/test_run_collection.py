from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig

from hydraflow.core.run_collection import Run, RunCollection


@dataclass
class Size:
    width: int = 0
    height: int = 0


@dataclass
class Config:
    count: int = 1
    name: str = "a"
    size: Size = field(default_factory=Size)


@pytest.fixture(scope="module")
def results(collect):
    file = Path(__file__).parent / "run.py"
    return collect(file, ["-m", "count=1,2", "size.width=10,20,30", "name=abc,def"])


@pytest.fixture(scope="module")
def run_dirs(results: list[tuple[Path, DictConfig]]):
    return [r[0].parent for r in results]


def test_run_dirs_len(run_dirs: list[Path]):
    assert len(run_dirs) == 12


@pytest.fixture(scope="module")
def runs(run_dirs: list[Path]):
    return [Run(run_dir) for run_dir in run_dirs]


@pytest.fixture(scope="module")
def rc(runs: list[Run]):
    return RunCollection(runs)


def test_len(rc: RunCollection[Run[Config]]):
    assert len(rc) == 12


def test_bool(rc: RunCollection[Run[Config]]):
    assert bool(rc) is True


def test_getitem_int(rc: RunCollection[Run[Config]]):
    assert isinstance(rc[0], Run)


def test_getitem_slice(rc: RunCollection[Run[Config]]):
    assert isinstance(rc[:3], RunCollection)


def test_getitem_iterable(rc: RunCollection[Run[Config]]):
    assert isinstance(rc[[0, 1, 2]], RunCollection)


def test_getitem_iter(rc: RunCollection[Run[Config]]):
    assert len(list(rc)) == 12


def test_filter(rc: RunCollection[Run[Config]]):
    assert len(rc.filter(count=1)) == 6


def test_filter_callable(rc: RunCollection[Run[Config]]):
    assert len(rc.filter(lambda r: r.cfg.count == 1)) == 6


def test_filter_dict(rc: RunCollection[Run[Config]]):
    assert len(rc.filter({"size.width": 10})) == 4


def test_filter_dict_list(rc: RunCollection[Run[Config]]):
    assert len(rc.filter({"size.width": [10, 30]})) == 8


def test_filter_dict_tuple(rc: RunCollection[Run[Config]]):
    assert len(rc.filter({"size.width": (20, 30)})) == 8


def test_filter_multi(rc: RunCollection[Run[Config]]):
    assert len(rc.filter({"size.width": (20, 30)}, count=1, name="abc")) == 2


def test_try_get(rc: RunCollection[Run[Config]]):
    assert rc.try_get({"size.height": 10}) is None


def test_get(rc: RunCollection[Run[Config]]):
    r = rc.get({"size.width": 10}, count=1, name="abc")
    assert r.cfg.count == 1
    assert r.cfg.name == "abc"
    assert r.cfg.size.width == 10


def test_to_list(rc: RunCollection[Run[Config]]):
    assert sorted(rc.to_list("name")) == [*(["abc"] * 6), *(["def"] * 6)]


# @pytest.fixture(scope="module")
# def rc(results: list[tuple[Path, DictConfig]]):
#     return results.filter(lambda r: r.cfg.name == "abc")


# class Db:
#     name: str
#     b: int


# class Config:
#     a: int
#     db: Db


# @pytest.fixture
# def rc(root_dirs: list[str]):
#     return RunConfig[Config].load(root_dirs)


# def test_run_config_len(rc: RunCollection[RunConfig[Config]]):
#     assert len(rc) == 5


# def test_run_config_cfg(rc: RunCollection[RunConfig[Config]]):
#     assert all(not r.cfg for r in rc)


# def test_group_by_list_config(rc: RunCollection[RunConfig[Config]]):
#     rc.set_default("a", lambda x: list(x.script.grid_size))
#     gp = rc.group_by("a")
#     assert list(gp.keys()) == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]


# def test_group_by_list(rc: RunCollection[RunConfig[Config]]):
#     for r in rc:
#         r.dummy = [1, 2]  # type: ignore
#     gp = rc.group_by("dummy")
#     assert list(gp.keys()) == [(1, 2)]


# def test_group_by_numpy(rc: RunCollection[RunConfig[Config]]):
#     for r in rc:
#         r.dummy = np.array([1, 2])  # type: ignore
#     gp = rc.group_by("dummy")
#     assert list(gp.keys()) == [(1, 2)]


# def test_run_config_set_defaults(rc: RunCollection[RunConfig[Config]]):
#     rc.set_default("db.name", "abc")
#     assert all(r.cfg.db.name == "abc" for r in rc)
#     rc.set_default("a", 10)
#     assert all(r.cfg.a == 10 for r in rc)


# def test_run_config_set_defaults_callable(rc: RunCollection[RunConfig[Config]]):
#     rc.set_default("a", lambda x: x.script.grid_size[0])
#     assert rc.to_list("a") == [1, 2, 3, 1, 1]
#     rc.set_default("db.b", lambda x: x.script.grid_size[1])
#     assert rc.to_list("db.b") == [2, 3, 4, 2, 2]


# def test_run_config_set_defaults_tuple_callable(rc: RunCollection[RunConfig[Config]]):
#     rc.set_default(("a", "db.b"), lambda x: x.script.grid_size[:2])
#     assert rc.to_list("a") == [1, 2, 3, 1, 1]
#     assert rc.to_list("db.b") == [2, 3, 4, 2, 2]
#     rc.set_default(("a", "db.b"), lambda x: x.script.cell_size[:2])
#     assert rc.to_list("a") == [1, 2, 3, 1, 1]
#     assert rc.to_list("db.b") == [2, 3, 4, 2, 2]
#     for r in rc[:2]:
#         r.cfg.a = None  # type: ignore
#     for r in rc[3:]:
#         r.cfg.db.b = None  # type: ignore
#     rc.set_default(("a", "db.b"), lambda x: x.script.grid_size[1:])
#     assert rc.to_list("a") == [2, 3, 3, 1, 1]
#     assert rc.to_list("db.b") == [2, 3, 4, 3, 3]


# def test_run_series_set_defaults_type_error(root_dirs: list[str]):
#     rs = RunSeries.load(root_dirs)
#     with pytest.raises(AttributeError):
#         rs.set_default("a", 3)  # type: ignore


def test_to_hashable_fallback_str():
    from hydraflow.core.run_collection import to_hashable

    class C:
        __hash__ = None  # type: ignore

        def __str__(self) -> str:
            return "abc"

        def __iter__(self):
            raise TypeError

    assert to_hashable(C()) == "abc"
