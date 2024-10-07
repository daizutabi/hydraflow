import json
from dataclasses import dataclass, field

import pytest
from omegaconf import ListConfig, OmegaConf


def test_is_param_with_simple_values():
    from hydraflow.config import _is_param

    assert _is_param(1) is True
    assert _is_param("string") is True
    assert _is_param(3.14) is True
    assert _is_param(True) is True


def test_is_param_with_dictconfig_containing_simple_values():
    from hydraflow.config import _is_param

    dict_conf = OmegaConf.create({"a": 1, "b": "string", "c": 3.14, "d": True})
    assert _is_param(dict_conf) is False


def test_is_param_with_listconfig_containing_simple_values():
    from hydraflow.config import _is_param

    list_conf = OmegaConf.create([1, "string", 3.14, True])
    assert _is_param(list_conf) is True


def test_is_param_with_listconfig_containing_nested_dictconfig():
    from hydraflow.config import _is_param

    nested_list_conf = OmegaConf.create([1, {"a": 1}, 3.14])
    assert _is_param(nested_list_conf) is False


def test_is_param_with_listconfig_containing_nested_listconfig():
    from hydraflow.config import _is_param

    nested_list_conf_2 = OmegaConf.create([1, [2, 3], 3.14])
    assert _is_param(nested_list_conf_2) is False


def test_is_param_with_empty_dictconfig():
    from hydraflow.config import _is_param

    empty_dict_conf = OmegaConf.create({})
    assert _is_param(empty_dict_conf) is False


def test_is_param_with_empty_listconfig():
    from hydraflow.config import _is_param

    empty_list_conf = OmegaConf.create([])
    assert _is_param(empty_list_conf) is True


def test_is_param_with_none():
    from hydraflow.config import _is_param

    assert _is_param(None) is True


def test_is_param_with_complex_nested_structure():
    from hydraflow.config import _is_param

    complex_conf = OmegaConf.create({"a": [1, {"b": 2}], "c": {"d": 3}})
    assert _is_param(complex_conf) is False


def test_iter_params_with_none():
    from hydraflow.config import iter_params

    assert not list(iter_params(None))


def test_iter_params():
    from hydraflow.config import iter_params

    conf = OmegaConf.create({"k": "v", "l": [1, {"a": "1", "b": "2", 3: "c"}]})
    it = iter_params(conf)
    assert next(it) == ("k", "v")
    assert next(it) == ("l.0", 1)
    assert next(it) == ("l.1.a", "1")
    assert next(it) == ("l.1.b", "2")
    assert next(it) == ("l.1.3", "c")


def test_collect_params():
    from hydraflow.config import collect_params

    conf = OmegaConf.create({"k": "v", "l": [1, {"a": "1", "b": "2", 3: "c"}]})
    params = collect_params(conf)
    assert params == {"k": "v", "l.0": 1, "l.1.a": "1", "l.1.b": "2", "l.1.3": "c"}


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
    assert next(it) == ("store.items", ["a", "b"])


def test_iter_params_with_empty_config():
    from hydraflow.config import iter_params

    empty_cfg = Config(
        size=Size(x=0, y=0),
        db=Db(name="", port=0),
        store=Store(items=[]),
    )
    it = iter_params(empty_cfg)
    assert next(it) == ("size.x", 0)
    assert next(it) == ("size.y", 0)
    assert next(it) == ("db.name", "")
    assert next(it) == ("db.port", 0)
    assert next(it) == ("store.items", [])


def test_iter_params_with_nested_config():
    from hydraflow.config import iter_params

    @dataclass
    class Nested:
        level1: Config = field(default_factory=Config)

    nested_cfg = Nested()
    it = iter_params(nested_cfg)
    assert next(it) == ("level1.size.x", 1)
    assert next(it) == ("level1.size.y", 2)
    assert next(it) == ("level1.db.name", "name")
    assert next(it) == ("level1.db.port", 100)
    assert next(it) == ("level1.store.items", ["a", "b"])


def test_iter_params_with_mixed_types_in_list():
    from hydraflow.config import iter_params

    @dataclass
    class MixedStore:
        items: list = field(default_factory=lambda: ["a", 1, {"key": "value"}])

    mixed_cfg = MixedStore()
    it = iter_params(mixed_cfg)
    assert next(it) == ("items.0", "a")
    assert next(it) == ("items.1", 1)
    assert next(it) == ("items.2.key", "value")


@pytest.mark.parametrize("type_", [int, float])
@pytest.mark.parametrize("s", ["[1, 2, 3]", "[1.0, 2.0, 3.0]"])
def test_list_config(type_, s):
    a = [type_(x) for x in [1, 2, 3]]
    b = OmegaConf.create(a)
    assert isinstance(b, ListConfig)
    t = OmegaConf.create(json.loads(s))
    assert b == t
    assert a == t


@pytest.mark.parametrize("s", ['["a", "b", "c"]'])
def test_list_config_str(s):
    a = ["a", "b", "c"]
    b = OmegaConf.create(a)
    assert isinstance(b, ListConfig)
    t = OmegaConf.create(json.loads(s))
    assert b == t


@pytest.mark.parametrize("x", [{"a": 1}, {"a": [1, 2, 3]}])
def test_collect_params_dict(x):
    from hydraflow.config import collect_params

    assert collect_params(x) == x


def test_collect_params_dict_dot():
    from hydraflow.config import collect_params

    assert collect_params({"a": {"b": 1}}) == {"a.b": 1}
    assert collect_params({"a.b": 1}) == {"a.b": 1}


def test_collect_params_list_dot():
    from hydraflow.config import collect_params

    assert collect_params(["a=1"]) == {"a": "1"}
    assert collect_params(["a.b=2", "c"]) == {"a.b": "2"}


@dataclass
class C:
    z: int = 3


@dataclass
class B:
    y: int = 2
    c: C = field(default_factory=C)


@dataclass
class A:
    x: int = 1
    b: B = field(default_factory=B)


def test_select_config():
    from hydraflow.config import select_config

    a = A()
    assert select_config(a, ["x"]) == {"x": 1}
    assert select_config(a, ["b.y"]) == {"b.y": 2}
    assert select_config(a, ["b.c.z"]) == {"b.c.z": 3}
    assert select_config(a, ["b.c.z", "x"]) == {"b.c.z": 3, "x": 1}
    assert select_config(a, ["b.c.z", "b.y"]) == {"b.c.z": 3, "b.y": 2}
