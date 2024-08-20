from dataclasses import dataclass, field

import pytest
from omegaconf import OmegaConf


def test_iter_params():
    from hydraflow.config import iter_params

    conf = OmegaConf.create({"k": "v", "l": [1, {"a": "1", "b": "2", 3: "c"}]})
    it = iter_params(conf)
    assert next(it) == ("k", "v")
    assert next(it) == ("l.0", 1)
    assert next(it) == ("l.1.a", "1")
    assert next(it) == ("l.1.b", "2")
    assert next(it) == ("l.1.3", "c")


@dataclass
class Size:
    x: int = 1
    y: int = 2


@dataclass
class Db:
    name: str = "name"
    port: int = 100


@dataclass
class Store:
    items: list[str] = field(default_factory=lambda: ["a", "b"])


@dataclass
class Config:
    size: Size = field(default_factory=Size)
    db: Db = field(default_factory=Db)
    store: Store = field(default_factory=Store)


@pytest.fixture
def cfg():
    return Config()


def test_config(cfg: Config):
    assert cfg.size.x == 1
    assert cfg.db.name == "name"
    assert cfg.store.items == ["a", "b"]


def test_iter_params_from_config(cfg):
    from hydraflow.config import iter_params

    it = iter_params(cfg)
    assert next(it) == ("size.x", 1)
    assert next(it) == ("size.y", 2)
    assert next(it) == ("db.name", "name")
    assert next(it) == ("db.port", 100)
    assert next(it) == ("store.items.0", "a")
    assert next(it) == ("store.items.1", "b")
