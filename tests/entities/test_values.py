from pathlib import Path

import pytest

from hydraflow.entities.run_collection import RunCollection

pytestmark = pytest.mark.xdist_group(name="group7")


@pytest.fixture(scope="module")
def rc(collect):
    file = Path(__file__).parent / "values.py"
    return collect(file, ["-m", "host=a", "x=1e-6,1e-8,1e-7"])


def test_rc_len(rc: RunCollection):
    assert len(rc) == 3


@pytest.mark.parametrize(
    ("names", "values"),
    [
        ("x", [1e-6, 1e-8, 1e-7]),
        ("y", [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
        (["host", "port"], [("a", 3306), ("a", 3306), ("a", 3306)]),
    ],
)
def test_values(rc: RunCollection, names, values):
    assert rc.values(names) == values


def test_sorted(rc: RunCollection):
    assert rc.sorted("x").values("x") == [1e-8, 1e-7, 1e-6]


def test_sorted_reverse(rc: RunCollection):
    assert rc.sorted("x", reverse=True).values("x") == [1e-6, 1e-7, 1e-8]
