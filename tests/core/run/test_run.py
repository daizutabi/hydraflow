from pathlib import Path

import pytest
from omegaconf import ListConfig

from hydraflow.core.run import Run
from hydraflow.core.run_collection import RunCollection


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


def test_get_default(run: Run[Config]):
    assert run.get("unknown", 10) == 10


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


def test_load():
    run = Run[Config].load("a/b/c")
    assert isinstance(run, Run)
    assert run.impl is None


@pytest.mark.parametrize("n_jobs", [0, 1, 2])
def test_load_collection(n_jobs: int):
    rc = Run[Config].load([Path("a/b/c"), Path("a/b/d")], n_jobs=n_jobs)
    assert isinstance(rc, RunCollection)
    assert len(rc) == 2
    assert rc[0].impl is None
    assert rc[1].impl is None


def test_load_impl():
    run = Run[Config, Impl].load("a/b/c", Impl)
    assert run.impl.path == Path("a/b/c/artifacts")


@pytest.mark.parametrize("n_jobs", [0, 1, 2])
def test_load_impl_collection(n_jobs: int):
    rc = Run[Config, Impl].load([Path("a/b/c"), Path("a/b/d")], Impl, n_jobs=n_jobs)
    assert isinstance(rc, RunCollection)
    assert len(rc) == 2
    assert rc[0].impl.path == Path("a/b/c/artifacts")
    assert rc[1].impl.path == Path("a/b/d/artifacts")


def test_impl_collection_repr():
    rc = Run[Config, Impl].load(["a/b/c", "a/b/d"], Impl)
    assert repr(rc) == "RunCollection(Run[Impl], n=2)"


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


class Dummy:
    pass


class ImplConfig:
    path: Path
    cfg: Dummy

    def __init__(self, path: Path, cfg: Dummy):
        self.path = path
        self.cfg = cfg


@pytest.fixture(scope="module")
def run_impl_config(results):
    run_dir: Path = results[0][0].parent
    return Run[Dummy, ImplConfig].load(run_dir, ImplConfig)


def test_impl_config_repr(run_impl_config: Run[Dummy, ImplConfig]):
    assert repr(run_impl_config).startswith("Run[ImplConfig]('")


def test_impl_config(run_impl_config: Run[Dummy, ImplConfig]):
    assert run_impl_config.impl.path.stem == "artifacts"
    cfg = run_impl_config.cfg
    assert cfg.count == 10  # type: ignore
    assert cfg.name == "abc"  # type: ignore
    assert cfg.size.width == 1  # type: ignore
    assert cfg.size.height == 3  # type: ignore
