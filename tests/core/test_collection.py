from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Self, override

import numpy as np
import polars as pl
import pytest
from omegaconf import ListConfig

from hydraflow.core.collection import Collection, matches, to_hashable
from hydraflow.core.group_by import GroupBy

if TYPE_CHECKING:
    from collections.abc import Callable

# pyright: reportPrivateUsage=false
# pyright: reportUnknownLambdaType=false
# pyright: reportUnknownArgumentType=false


@dataclass
class Config:
    x: int
    y: str


class Run[C]:
    cfg: C

    def __init__(self, cfg: C) -> None:
        self.cfg = cfg

    def get(self, key: str, default: Any | Callable[[Self], Any] | None = None) -> Any:
        if key == "run_id":
            return 0
        value = getattr(self.cfg, key, None)
        if value is not None:
            return value
        if callable(default):
            return default(self)
        return default


@pytest.fixture(scope="module")
def rc():
    x = [1, 2, 3]
    y = ["a", "b", "c", "d"]
    items = [Run[Config](Config(x, y)) for x, y in product(x, y)]
    return Collection(items, Run[Config].get)


type Rc = Collection[Run[Config]]


def test_repr(rc: Rc) -> None:
    x = repr(rc)
    assert x.startswith("Collection(")
    assert x.endswith(", n=12)")


def test_repr_empty() -> None:
    x: list[int] = []
    assert repr(Collection(x)) == "Collection(empty)"


def test_len(rc: Rc) -> None:
    assert len(rc) == 12


def test_bool(rc: Rc) -> None:
    assert bool(rc) is True


def test_getitem_int(rc: Rc) -> None:
    assert isinstance(rc[0], Run)


def test_getitem_slice(rc: Rc) -> None:
    rc = rc[:3]
    assert isinstance(rc, Collection)
    assert len(rc) == 3
    assert rc._get(rc[0], "x", None) == 1


def test_getitem_iterable(rc: Rc) -> None:
    rc = rc[[2, 3]]
    assert isinstance(rc, Collection)
    assert len(rc) == 2
    assert rc._get(rc[0], "y", None) == "c"


def test_getitem_str(rc: Rc) -> None:
    assert isinstance(rc["x"], pl.Series)


def test_iter(rc: Rc) -> None:
    assert len(list(iter(rc))) == 12


def test_filter(rc: Rc) -> None:
    rc = rc.filter(x=1)
    assert len(rc) == 4
    assert rc._get(rc[0], "x", None) == 1
    assert all(r.get("x") == 1 for r in rc)


def test_filter_multi(rc: Rc) -> None:
    rc = rc.filter(x=3, y="c")
    assert len(rc) == 1
    assert rc[0].get("x") == 3
    assert rc[0].get("y") == "c"


def test_filter_callable(rc: Rc) -> None:
    assert len(rc.filter(lambda r: r.get("x") >= 2)) == 8


def test_filter_tuple(rc: Rc) -> None:
    assert len(rc.filter(("x", (1, 2)), ("y", ["b", "c"]))) == 4
    assert len(rc.filter(("x", (1, 3)), ("y", ["b", "c"]))) == 6
    assert len(rc.filter(("x", (1, 2)), ("y", ["a", "c"]))) == 4


def test_filter_iterable(rc: Rc) -> None:
    rc = rc.filter(rc["x"] >= 2, rc["y"].is_in(["a", "b"]))
    assert len(rc) == 4
    assert all(rc["x"] >= 2)
    assert all(rc["y"].is_in(["a", "b"]))


def test_filter_complex(rc: Rc) -> None:
    rc = rc.filter(rc["x"] <= 2, ("x", (1, 3)), rc["y"] > "b", y=["a", "b", "d"])
    assert rc["x"].to_list() == [1, 2]
    assert rc["y"].to_list() == ["d", "d"]


def test_filter_none(rc: Rc) -> None:
    rc_ = rc.filter()
    assert list(rc_) == list(rc)


def test_exclude(rc: Rc) -> None:
    rc = rc.exclude(rc["x"] <= 2, y=["a", "b", "d"])
    assert rc["x"].to_list() == [3]
    assert rc["y"].to_list() == ["c"]


def test_exclude_none(rc: Rc) -> None:
    rc_ = rc.exclude()
    assert list(rc_) == list(rc)


def test_try_get(rc: Rc) -> None:
    assert rc.try_get(("x", 10)) is None


def test_try_get_error(rc: Rc) -> None:
    with pytest.raises(ValueError):
        rc.try_get(x=1)


def test_get(rc: Rc) -> None:
    r = rc.get(x=1, y="a")
    assert r.get("x") == 1
    assert r.get("y") == "a"


def test_get_error(rc: Rc) -> None:
    with pytest.raises(ValueError):
        rc.get(x=100)


def test_first(rc: Rc) -> None:
    r = rc.first(x=2)
    assert r.get("x") == 2
    assert r.get("y") == "a"


def test_first_error(rc: Rc) -> None:
    with pytest.raises(ValueError):
        rc.first(x=100)


def test_last(rc: Rc) -> None:
    r = rc.last(x=3)
    assert r.get("x") == 3
    assert r.get("y") == "d"


def test_last_error(rc: Rc) -> None:
    with pytest.raises(ValueError):
        rc.last(x=100)


def test_to_list(rc: Rc) -> None:
    assert sorted(rc.to_list("x")) == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]


def test_to_list_default(rc: Rc) -> None:
    assert sorted(rc.to_list("unknown", 1)) == [1] * 12


def test_to_list_default_callable(rc: Rc) -> None:
    x = rc.to_list("unknown", lambda run: run.cfg.x + 1)
    assert x == [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]


def test_to_numpy(rc: Rc) -> None:
    assert np.array_equal(rc.to_numpy("x")[3:5], [1, 2])


def test_to_series(rc: Rc) -> None:
    s = rc.to_series("x", name="X")
    assert s.to_list() == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    assert s.name == "X"


def test_unique(rc: Rc) -> None:
    assert np.array_equal(rc.unique("x"), [1, 2, 3])


def test_n_unique(rc: Rc) -> None:
    assert rc.n_unique("y") == 4


def test_sort(rc: Rc) -> None:
    x = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    assert rc.sort("x", reverse=True).to_list("x") == x[::-1]


def test_sort_multi(rc: Rc) -> None:
    x = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    assert rc.sort("y", "x").to_list("x") == x


def test_sort_emtpy(rc: Rc) -> None:
    assert rc.sort().to_list("x")[-1] == 3


def test_to_frame_empty(rc: Rc) -> None:
    df = rc.to_frame()
    assert df.shape == (0, 0)


def test_to_frame(rc: Rc) -> None:
    df = rc.to_frame("x", "y", "run_id")
    assert df.shape == (12, 3)
    assert df.columns == ["x", "y", "run_id"]
    assert df.item(0, "x") == 1
    assert df.item(0, "y") == "a"
    assert df.item(0, "run_id") == 0
    assert df.item(-1, "x") == 3
    assert df.item(-1, "y") == "d"
    assert df.item(-1, "run_id") == 0


def test_to_frame_defaults(rc: Rc) -> None:
    df = rc.to_frame("x", "z", defaults={"z": lambda r: r.cfg.x + 2})
    assert df.shape == (12, 2)
    assert df.columns == ["x", "z"]
    assert df.item(0, "x") == 1
    assert df.item(0, "z") == 3
    assert df.item(-1, "x") == 3
    assert df.item(-1, "z") == 5


def test_to_frame_tuple(rc: Rc) -> None:
    df = rc.to_frame("x", ("z", lambda r: r.cfg.x + 3))
    assert df.shape == (12, 2)
    assert df.columns == ["x", "z"]
    assert df.item(0, "x") == 1
    assert df.item(0, "z") == 4
    assert df.item(-1, "x") == 3
    assert df.item(-1, "z") == 6


def test_to_frame_kwargs(rc: Rc) -> None:
    df = rc.to_frame("x", "y", "run_id", z=lambda r: r.cfg.x + 1)
    assert df.shape == (12, 4)
    assert df.columns == ["x", "y", "run_id", "z"]
    assert df.item(0, "z") == 2
    assert df.item(-1, "z") == 4


def test_to_frame_kwargs_without_keys(rc: Rc) -> None:
    df = rc.to_frame(z=lambda r: r.cfg.x + 1)
    assert df.shape == (12, 1)
    assert df.columns == ["z"]
    assert df.item(0, "z") == 2
    assert df.item(-1, "z") == 4


def test_group_by(rc: Rc) -> None:
    gp = rc.group_by("y")
    assert isinstance(gp, GroupBy)
    assert len(gp) == 4
    assert len(gp["a"]) == 3
    rc = gp["b"]
    assert rc._get(rc[0], "x", None) == 1
    assert rc._get(rc[0], "y", None) == "b"


def test_group_by_multi(rc: Rc) -> None:
    gp = rc.group_by("x", "run_id")
    assert isinstance(gp, GroupBy)
    assert len(gp) == 3
    assert len(gp[1, 0]) == 4
    rc = gp[3, 0]
    assert rc._get(rc[0], "x", None) == 3
    assert rc._get(rc[0], "run_id", None) == 0


def test_to_hashable_list_config() -> None:
    assert to_hashable(ListConfig([1, 2, 3])) == (1, 2, 3)


def test_to_hashable_ndarray() -> None:
    assert to_hashable(np.array([1, 2, 3])) == (1, 2, 3)


def test_to_hashable_fallback_str() -> None:
    class C:
        __hash__ = None  # pyright: ignore[reportAssignmentType, reportUnannotatedClassAttribute]

        @override
        def __str__(self) -> str:
            return "abc"

        def __iter__(self):
            raise TypeError

    assert to_hashable(C()) == "abc"


@pytest.mark.parametrize(
    ("criterion", "expected"),
    [
        (10, True),
        (1, False),
        ([20, 10], True),
        ([1, 2], False),
        ((1, 10), True),
        ((10, 1), False),
        (ListConfig([10, 20]), True),
        (lambda x: x == 10, True),
        (lambda x: x > 10, False),
    ],
)
def test_matches(criterion: Any, *, expected: bool) -> None:
    assert matches(10, criterion) is expected


def test_matches_list_config() -> None:
    assert matches(ListConfig([10, 20]), [10, 20])
    assert matches(ListConfig([10, 20]), ListConfig([10, 20]))


@pytest.mark.parametrize("seed", [None, 1])
def test_sample(seed: int | None) -> None:
    x = Collection(list(range(100)))

    sample = x.sample(50, seed=seed)
    assert isinstance(sample, Collection)
    assert len(sample) == 50
    assert len(set(sample._items)) == 50


def test_sample_error() -> None:
    x = Collection(list(range(10)))
    with pytest.raises(ValueError):
        x.sample(11)


@pytest.mark.parametrize("seed", [None, 1])
def test_shuffle(seed: int | None) -> None:
    x = Collection(list(range(10)))
    shuffled = x.shuffle(seed)
    assert len(shuffled) == 10
    assert len(set(shuffled._items)) == 10
    assert shuffled._items != x._items


@pytest.fixture(scope="module")
def rcd():
    x = [1, 2, 3, 4, 5]
    y = [1, 1, 3, 3, 3]
    z = ["abcd", "bac", "bacd", "abc", "abcd"]
    items = [{"x": x, "y": y, "z": z} for x, y, z in zip(x, y, z, strict=True)]
    return Collection(items, lambda i, k, d: i.get(k, d))


def test_eq(rcd: Collection[dict[str, int | str]]) -> None:
    assert len(rcd.filter(rcd.eq("x", "y"))) == 2


def test_ne(rcd: Collection[dict[str, int | str]]) -> None:
    assert len(rcd.filter(rcd.ne("x", "y"))) == 3


def test_gt(rcd: Collection[dict[str, int | str]]) -> None:
    assert len(rcd.filter(rcd.gt("x", "y"))) == 3


def test_lt(rcd: Collection[dict[str, int | str]]) -> None:
    assert len(rcd.filter(rcd.lt("x", "y"))) == 0


def test_ge(rcd: Collection[dict[str, int | str]]) -> None:
    assert len(rcd.filter(rcd.ge("x", "y"))) == 5


def test_le(rcd: Collection[dict[str, int | str]]) -> None:
    assert len(rcd.filter(rcd.le("x", "y"))) == 2


def test_startswith(rcd: Collection[dict[str, int | str]]) -> None:
    assert len(rcd.filter(rcd.startswith("z", "ab"))) == 3


def test_endswith(rcd: Collection[dict[str, int | str]]) -> None:
    assert len(rcd.filter(rcd.endswith("z", "cd"))) == 3


def test_match(rcd: Collection[dict[str, int | str]]) -> None:
    assert len(rcd.filter(rcd.match("z", r".*ac.*"))) == 2


def test_match_flags(rcd: Collection[dict[str, int | str]]) -> None:
    assert len(rcd.filter(rcd.match("z", r".*AC.*", flags=re.IGNORECASE))) == 2
