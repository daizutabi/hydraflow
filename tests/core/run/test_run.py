from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from omegaconf import ListConfig

from hydraflow.core.run import Run
from hydraflow.core.run_collection import RunCollection


class Db:
    name: str = ""
    b: int = 0


class Config:
    a: int = 0
    db: Db = Db()


@pytest.fixture
def run():
    return Run[Config](Path())


def test_repr(run: Run[Config]):
    assert repr(run) == "Run('')"


def test_update_str(run: Run[Config]):
    run.update("a", 10)
    assert run.get("a") == 10
    run.update("a", 20)
    assert run.get("a") == 10


def test_update_str_force(run: Run[Config]):
    run.update("a", 10)
    assert run.get("a") == 10
    run.update("a", 20, force=True)
    assert run.get("a") == 20


def test_update_str_dot(run: Run[Config]):
    run.update("db.name", "abc")
    assert run.get("db.name") == "abc"
    run.update("db.name", "def")
    assert run.get("db.name") == "abc"


def test_update_str_underscore(run: Run[Config]):
    run.update("db__name", "abc")
    assert run.get("db.name") == "abc"
    assert run.get("db__name") == "abc"
    run.update("db__name", "def")
    assert run.get("db.name") == "abc"
    assert run.get("db__name") == "abc"


def test_update_str_dot_force(run: Run[Config]):
    run.update("db.b", 100)
    assert run.get("db.b") == 100
    run.update("db.b", 200, force=True)
    assert run.get("db.b") == 200


def test_update_callable(run: Run[Config]):
    run.update("db.name", lambda _: "abc")
    run.update("db.b", lambda run: len(run.cfg.db.name))
    assert run.get("db.b") == 3
    run.update("db.b", lambda run: run.cfg.db.b * 10)
    assert run.get("db.b") == 3


def test_update_tuple(run: Run[Config]):
    run.update(("db.name", "db.b"), ["xyz", 1000])
    assert run.get("db.name") == "xyz"
    assert run.get("db.b") == 1000
    run.update(("db.name", "a"), ["abc", 1])
    assert run.get("db.name") == "xyz"
    assert run.get("a") == 1


def test_update_underscore(run: Run[Config]):
    run.update(("db__name", "db__b"), ["xyz", 1000])
    assert run.get("db.name") == "xyz"
    assert run.get("db.b") == 1000
    assert run.get("db__name") == "xyz"
    assert run.get("db__b") == 1000
    run.update(("db__name", "a"), ["abc", 1])
    assert run.get("db.name") == "xyz"
    assert run.get("db__name") == "xyz"
    assert run.get("a") == 1


def test_update_tuple_callable(run: Run[Config]):
    run.update(("db.name", "db.b"), lambda x: ["a", 1])
    assert run.get("db.name") == "a"
    assert run.get("db.b") == 1
    run.update(("db.name", "a"), lambda x: ["b", 2])
    assert run.get("db.name") == "a"
    assert run.get("a") == 2
    run.update(("db.name", "a"), lambda x: [1 / 0, 1 / 0])


def test_update_tuple_error(run: Run[Config]):
    with pytest.raises(TypeError):
        run.update(("db.name", "db.b"), lambda x: "ab")


def test_get_error(run: Run[Config]):
    with pytest.raises(AttributeError):
        run.get("unknown")


def test_get_default(run: Run[Config]):
    assert run.get("unknown", 10) == 10


def test_get_default_callable(run: Run[Config]):
    run.update("a", 1000)
    assert run.get("unknown", lambda run: run.get("a")) == 1000


def test_get_info(run: Run[Config]):
    assert run.get("run_dir").as_posix() == "."


def test_lit(run: Run[Config]):
    run.update("db.b", 100)
    expr = run.lit("db.b", dtype=pl.Int64)
    df = pl.DataFrame({"a": [1, 2, 3]})
    df = df.with_columns(expr)
    assert df.shape == (3, 2)
    assert df.columns == ["a", "db.b"]
    assert df.item(0, "db.b") == 100
    assert df.item(1, "db.b") == 100
    assert df.item(2, "db.b") == 100


def test_to_frame(run: Run[Config]):
    def func(run: Run[Config]) -> pl.DataFrame:
        return pl.DataFrame({"a": [run.get("a"), 20]})

    run.update("a", 10)
    run.update("db.b", 100)
    df = run.to_frame(func, "db.b", ("x", 123), ("y", lambda r: r.get("a")))
    assert df.shape == (2, 4)
    assert df["a"].to_list() == [10, 20]
    assert df["db.b"].to_list() == [100, 100]
    assert df["x"].to_list() == [123, 123]
    assert df["y"].to_list() == [10, 10]


def test_to_dict(run: Run[Config]):
    run.update("a", 10)
    run.update("db.name", "abc")
    run.update("db.b", 100)
    assert run.to_dict(flatten=True) == {
        "a": 10,
        "db.name": "abc",
        "db.b": 100,
    }


def test_to_dict_flatten_false(run: Run[Config]):
    run.update("a", 10)
    run.update("db.name", "abc")
    run.update("db.b", 100)
    assert run.to_dict(flatten=False) == {
        "a": 10,
        "db": {"name": "abc", "b": 100},
    }


def test_to_dict_error(run: Run[Config]):
    run.cfg = ListConfig([1, 2, 3])  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(TypeError):
        run.to_dict()


def test_impl_none(run: Run[Config]):
    assert run.impl is None


class Impl:
    path: str

    def __init__(self, path: Path):
        self.path = path.as_posix()


def test_impl():
    run = Run[Config, Impl](Path(), Impl)
    assert run.impl.path == "artifacts"


def test_repr_impl():
    run = Run[Config, Impl](Path("a/b/c"), Impl)
    assert repr(run) == "Run[Impl]('c')"


def test_load():
    run = Run[Config].load("a/b/c")
    assert isinstance(run, Run)
    assert run.impl is None


def test_get_cfg():
    run = Run[Config, Impl](Path(), Impl)
    assert run.get("cfg") is run.cfg


def test_get_impl():
    run = Run[Config, Impl](Path(), Impl)
    assert run.get("impl") is run.impl


@pytest.mark.parametrize("n_jobs", [0, 1, 2])
def test_load_collection(n_jobs: int):
    rc = Run[Config].load([Path("a/b/c"), Path("a/b/d")], n_jobs=n_jobs)
    assert isinstance(rc, RunCollection)
    assert len(rc) == 2
    assert rc[0].impl is None
    assert rc[1].impl is None


def test_load_impl():
    run = Run[Config, Impl].load("a/b/c", Impl)
    assert run.impl.path == "a/b/c/artifacts"


def test_get_impl_str():
    run = Run[Config, Impl].load("a/b/c", Impl)
    assert run.get("path") == "a/b/c/artifacts"
