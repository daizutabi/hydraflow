from dataclasses import dataclass, field

import pytest


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


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        (["x"], {"x": 1}),
        (["b.y"], {"b.y": 2}),
        (["b.c.z"], {"b.c.z": 3}),
        (["b.c.z", "x"], {"b.c.z": 3, "x": 1}),
        (["b.c.z", "b.y"], {"b.c.z": 3, "b.y": 2}),
    ],
)
def test_select_config(names, expected):
    from hydraflow.config import select_config

    a = A()
    assert select_config(a, names) == expected


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (["x=4"], {"x": 1}),
        (["b.y=2"], {"b.y": 2}),
        (["b.c.z=10"], {"b.c.z": 3}),
        (["b.c.z=1", "x=0"], {"b.c.z": 3, "x": 1}),
        (["b.c.z=1", "b.y=1"], {"b.c.z": 3, "b.y": 2}),
    ],
)
def test_select_overrides(overrides, expected):
    from hydraflow.config import select_overrides

    a = A()
    assert select_overrides(a, overrides) == expected
