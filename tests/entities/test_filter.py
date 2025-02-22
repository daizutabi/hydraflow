from pathlib import Path

import pytest
from pandas import DataFrame

from hydraflow.entities.run_collection import RunCollection

pytestmark = pytest.mark.xdist_group(name="group7")


@pytest.fixture(scope="module")
def rc(collect):
    file = Path(__file__).parent / "filter.py"
    collect(file, ["-m", "host=a,b", "port=1,2,3"])
    return collect(file, ["-m", "host=b,c", "port=1,2,4"])


def test_rc_len(rc: RunCollection):
    assert len(rc) == 10


@pytest.fixture(scope="module")
def df(rc: RunCollection):
    return rc.data.config.sort_values(["host", "port"])


@pytest.mark.parametrize(
    ("i", "host", "port"),
    [
        (0, "a", 1),
        (1, "a", 2),
        (2, "a", 3),
        (3, "b", 1),
        (4, "b", 2),
        (5, "b", 3),
        (6, "b", 4),
        (7, "c", 1),
        (8, "c", 2),
        (9, "c", 4),
    ],
)
def test_params(df: DataFrame, i: int, host: str, port: int):
    assert df.iloc[i]["host"] == host
    assert df.iloc[i]["port"] == port
