import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any, Self

import numpy as np
import pytest
from omegaconf import ListConfig

from hydraflow.core.collection import Collection

# pyright: reportPrivateUsage=false
# pyright: reportUnknownLambdaType=false
# pyright: reportUnknownArgumentType=false


@dataclass
class Config:
    x: int
    y: str


class Run[C]:
    cfg: C

    def __init__(self, cfg: C):
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


def test_repr(rc: Rc):
    x = repr(rc)
    assert x.startswith("Collection(")
    assert x.endswith(", n=12)")


def test_repr_empty():
    x: list[int] = []
    assert repr(Collection(x)) == "Collection(empty)"


def test_len(rc: Rc):
    assert len(rc) == 12


def test_bool(rc: Rc):
    assert bool(rc) is True


def test_getitem_int(rc: Rc):
    assert isinstance(rc[0], Run)


def test_getitem_slice(rc: Rc):
    rc = rc[:3]
    assert isinstance(rc, Collection)
    assert len(rc) == 3
    assert rc._get(rc[0], "x", None) == 1


def test_getitem_iterable(rc: Rc):
    rc = rc[[2, 3]]
    assert isinstance(rc, Collection)
    assert len(rc) == 2
    assert rc._get(rc[0], "y", None) == "c"


def test_iter(rc: Rc):
    assert len(list(iter(rc))) == 12


def test_filter(rc: Rc):
    rc = rc.filter(x=1)
    assert len(rc) == 4
    assert rc._get(rc[0], "x", None) == 1
    assert all(r.get("x") == 1 for r in rc)


def test_filter_multi(rc: Rc):
    rc = rc.filter(x=3, y="c")
    assert len(rc) == 1
    assert rc[0].get("x") == 3
    assert rc[0].get("y") == "c"


def test_filter_callable(rc: Rc):
    assert len(rc.filter(lambda r: r.get("x") >= 2)) == 8


def test_filter_tuple(rc: Rc):
    assert len(rc.filter(("x", (1, 2)), ("y", ["b", "c"]))) == 4
    assert len(rc.filter(("x", (1, 3)), ("y", ["b", "c"]))) == 6
    assert len(rc.filter(("x", (1, 2)), ("y", ["a", "c"]))) == 4


def test_try_get(rc: Rc):
    assert rc.try_get(("x", 10)) is None


def test_try_get_error(rc: Rc):
    with pytest.raises(ValueError):
        rc.try_get(x=1)


def test_get(rc: Rc):
    r = rc.get(x=1, y="a")
    assert r.get("x") == 1
    assert r.get("y") == "a"


def test_get_error(rc: Rc):
    with pytest.raises(ValueError):
        rc.get(x=100)


def test_first(rc: Rc):
    r = rc.first(x=2)
    assert r.get("x") == 2
    assert r.get("y") == "a"


def test_first_error(rc: Rc):
    with pytest.raises(ValueError):
        rc.first(x=100)


def test_last(rc: Rc):
    r = rc.last(x=3)
    assert r.get("x") == 3
    assert r.get("y") == "d"


def test_last_error(rc: Rc):
    with pytest.raises(ValueError):
        rc.last(x=100)


def test_to_list(rc: Rc):
    assert sorted(rc.to_list("x")) == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]


def test_to_list_default(rc: Rc):
    assert sorted(rc.to_list("unknown", 1)) == [1] * 12


def test_to_list_default_callable(rc: Rc):
    x = rc.to_list("unknown", lambda run: run.cfg.x + 1)
    assert x == [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]


def test_to_numpy(rc: Rc):
    assert np.array_equal(rc.to_numpy("x")[3:5], [1, 2])


def test_to_series(rc: Rc):
    s = rc.to_series("x", name="X")
    assert s.to_list() == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    assert s.name == "X"


def test_unique(rc: Rc):
    assert np.array_equal(rc.unique("x"), [1, 2, 3])


def test_n_unique(rc: Rc):
    assert rc.n_unique("y") == 4


def test_sort(rc: Rc):
    x = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    assert rc.sort("x", reverse=True).to_list("x") == x[::-1]


def test_sort_multi(rc: Rc):
    x = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    assert rc.sort("y", "x").to_list("x") == x


def test_sort_emtpy(rc: Rc):
    assert rc.sort().to_list("x")[-1] == 3


def test_to_frame_empty(rc: Rc):
    df = rc.to_frame()
    assert df.shape == (0, 0)


def test_to_frame(rc: Rc):
    df = rc.to_frame("x", "y", "run_id")
    assert df.shape == (12, 3)
    assert df.columns == ["x", "y", "run_id"]
    assert df.item(0, "x") == 1
    assert df.item(0, "y") == "a"
    assert df.item(0, "run_id") == 0
    assert df.item(-1, "x") == 3
    assert df.item(-1, "y") == "d"
    assert df.item(-1, "run_id") == 0


def test_to_frame_defaults(rc: Rc):
    df = rc.to_frame("x", "z", defaults={"z": lambda r: r.cfg.x + 2})
    assert df.shape == (12, 2)
    assert df.columns == ["x", "z"]
    assert df.item(0, "x") == 1
    assert df.item(0, "z") == 3
    assert df.item(-1, "x") == 3
    assert df.item(-1, "z") == 5


def test_to_frame_tuple(rc: Rc):
    df = rc.to_frame("x", ("z", lambda r: r.cfg.x + 3))
    assert df.shape == (12, 2)
    assert df.columns == ["x", "z"]
    assert df.item(0, "x") == 1
    assert df.item(0, "z") == 4
    assert df.item(-1, "x") == 3
    assert df.item(-1, "z") == 6


def test_to_frame_kwargs(rc: Rc):
    df = rc.to_frame("x", "y", "run_id", z=lambda r: r.cfg.x + 1)
    assert df.shape == (12, 4)
    assert df.columns == ["x", "y", "run_id", "z"]
    assert df.item(0, "z") == 2
    assert df.item(-1, "z") == 4


def test_to_frame_kwargs_without_keys(rc: Rc):
    df = rc.to_frame(z=lambda r: r.cfg.x + 1)
    assert df.shape == (12, 1)
    assert df.columns == ["z"]
    assert df.item(0, "z") == 2
    assert df.item(-1, "z") == 4


@pytest.mark.parametrize("progress", [False, True])
def test_to_frame_parallel(rc: Rc, progress: bool):
    df = rc.to_frame(
        z=lambda r: r.cfg.x + 10,
        n_jobs=2,
        backend="threading",
        progress=progress,
    )
    assert df.shape == (12, 1)
    assert df.columns == ["z"]
    assert df.item(0, "z") == 11
    assert df.item(-1, "z") == 13


def test_group_by(rc: Rc):
    from hydraflow.core.group_by import GroupBy

    gp = rc.group_by("y")
    assert isinstance(gp, GroupBy)
    assert len(gp) == 4
    assert len(gp["a"]) == 3
    rc = gp["b"]
    assert rc._get(rc[0], "x", None) == 1
    assert rc._get(rc[0], "y", None) == "b"


def test_group_by_multi(rc: Rc):
    from hydraflow.core.group_by import GroupBy

    gp = rc.group_by("x", "run_id")
    assert isinstance(gp, GroupBy)
    assert len(gp) == 3
    assert len(gp[1, 0]) == 4
    rc = gp[3, 0]
    assert rc._get(rc[0], "x", None) == 3
    assert rc._get(rc[0], "run_id", None) == 0


def test_to_hashable_list_config():
    from hydraflow.core.collection import to_hashable

    assert to_hashable(ListConfig([1, 2, 3])) == (1, 2, 3)


def test_to_hashable_ndarray():
    from hydraflow.core.collection import to_hashable

    assert to_hashable(np.array([1, 2, 3])) == (1, 2, 3)


def test_to_hashable_fallback_str():
    from hydraflow.core.collection import to_hashable

    class C:
        __hash__ = None  # pyright: ignore[reportAssignmentType, reportUnannotatedClassAttribute]

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
def test_matches(criterion: Any, expected: bool):
    from hydraflow.core.collection import matches

    assert matches(10, criterion) is expected


def test_matches_list_config():
    from hydraflow.core.collection import matches

    assert matches(ListConfig([10, 20]), [10, 20])
    assert matches(ListConfig([10, 20]), ListConfig([10, 20]))


@pytest.mark.parametrize("seed", [None, 1])
def test_sample(seed: int | None):
    from hydraflow.core.collection import Collection

    x = Collection(list(range(100)))

    sample = x.sample(50, seed=seed)
    assert isinstance(sample, Collection)
    assert len(sample) == 50
    assert len(set(sample._items)) == 50


def test_sample_error():
    from hydraflow.core.collection import Collection

    x = Collection(list(range(10)))
    with pytest.raises(ValueError):
        x.sample(11)


@pytest.mark.parametrize("seed", [None, 1])
def test_shuffle(seed: int | None):
    from hydraflow.core.collection import Collection

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


def test_eq(rcd: Collection[dict[str, int | str]]):
    assert len(rcd.filter(rcd.eq("x", "y"))) == 2


def test_ne(rcd: Collection[dict[str, int | str]]):
    assert len(rcd.filter(rcd.ne("x", "y"))) == 3


def test_gt(rcd: Collection[dict[str, int | str]]):
    assert len(rcd.filter(rcd.gt("x", "y"))) == 3


def test_lt(rcd: Collection[dict[str, int | str]]):
    assert len(rcd.filter(rcd.lt("x", "y"))) == 0


def test_ge(rcd: Collection[dict[str, int | str]]):
    assert len(rcd.filter(rcd.ge("x", "y"))) == 5


def test_le(rcd: Collection[dict[str, int | str]]):
    assert len(rcd.filter(rcd.le("x", "y"))) == 2


def test_startswith(rcd: Collection[dict[str, int | str]]):
    assert len(rcd.filter(rcd.startswith("z", "ab"))) == 3


def test_endswith(rcd: Collection[dict[str, int | str]]):
    assert len(rcd.filter(rcd.endswith("z", "cd"))) == 3


def test_match(rcd: Collection[dict[str, int | str]]):
    assert len(rcd.filter(rcd.match("z", r".*ac.*"))) == 2


def test_match_flags(rcd: Collection[dict[str, int | str]]):
    assert len(rcd.filter(rcd.match("z", r".*AC.*", flags=re.IGNORECASE))) == 2
