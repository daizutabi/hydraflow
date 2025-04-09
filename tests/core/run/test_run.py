from pathlib import Path

import pytest
from omegaconf import ListConfig

from hydraflow.core.run import Run


class Db:
    name: str
    b: int


class Config:
    a: int
    db: Db


@pytest.fixture
def run():
    return Run(Path())


def test_repr(run: Run):
    assert repr(run) == "Run('')"


def test_update_str(run: Run):
    run.update("a", 10)
    assert run.get("a") == 10
    run.update("a", 20)
    assert run.get("a") == 10


def test_update_str_force(run: Run):
    run.update("a", 10)
    assert run.get("a") == 10
    run.update("a", 20, force=True)
    assert run.get("a") == 20


def test_update_str_dot(run: Run):
    run.update("db.name", "abc")
    assert run.get("db.name") == "abc"
    run.update("db.name", "def")
    assert run.get("db.name") == "abc"


def test_update_str_dot_force(run: Run):
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


def test_get_info(run: Run[Config]):
    assert run.get("run_dir") == "."


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (10, True),
        (1, False),
        ([20, 10], True),
        ([1, 2], False),
        ((1, 10), True),
        ((10, 1), False),
        (ListConfig([10, 20]), True),
    ],
)
def test_predicate(run: Run[Config], value, expected):
    run.update("a", 10)
    assert run.predicate("a", value) is expected


def test_predicate_list_config(run: Run[Config]):
    run.update("a", ListConfig([10, 20]))
    assert run.predicate("a", [10, 20]) is True
    assert run.predicate("a", ListConfig([10, 20])) is True


def test_predicate_callable(run: Run[Config]):
    run.update("a", 10)
    assert run.predicate("a", lambda x: x > 5) is True
    assert run.predicate("a", lambda x: x > 15) is False


def test_predicate_tuple(run: Run[Config]):
    run.update("a", (1, 2))
    assert run.predicate("a", (1, 2)) is True
    assert run.predicate("a", (2, 1)) is False


def test_to_dict(run: Run[Config]):
    run.update("a", 10)
    run.update("db.name", "abc")
    run.update("db.b", 100)
    assert run.to_dict() == {
        "run_id": "",
        "run_dir": ".",
        "job_name": "",
        "a": 10,
        "db.name": "abc",
        "db.b": 100,
    }


def test_impl_none(run: Run[Config]):
    assert run.impl is None


class Impl:
    path: Path

    def __init__(self, path: Path):
        self.path = path


def test_impl():
    run = Run[Config, Impl](Path(), Impl)
    assert run.impl.path == Path("artifacts")


def test_repr_impl():
    run = Run[Config, Impl](Path("a/b/c"), Impl)
    assert repr(run) == "Run[Impl]('c')"


@pytest.fixture(scope="module")
def results(collect):
    file = Path(__file__).parent / "run.py"
    return collect(file, ["count=10", "name=abc", "size.width=1", "size.height=3"])


def test_len(results):
    assert len(results) == 1


def test_config(results):
    run_dir = results[0][0].parent
    run = Run(run_dir)
    assert run.get("count") == 10
    assert run.get("name") == "abc"
    assert run.get("size.width") == 1
    assert run.get("size.height") == 3
    assert run.get("size") == {"width": 1, "height": 3}
