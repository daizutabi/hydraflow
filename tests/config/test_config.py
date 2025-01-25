from dataclasses import dataclass, field


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
