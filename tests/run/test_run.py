from typing import TYPE_CHECKING

import pytest

from hydraflow.run_collection import RunCollection

if TYPE_CHECKING:
    from .run import Config

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(scope="module")
def rc(collect):
    args = ["-m", "port=2,1"]
    return collect("run/run.py", args)


def test_rc_len(rc: RunCollection):
    assert len(rc) == 2


def test_map_config_int(rc: RunCollection):
    assert list(rc.map_config(lambda cfg: cfg.port)) == [2, 1]


def test_map_config_list(rc: RunCollection):
    assert list(rc.map_config(lambda cfg: cfg.data.x)) == [[1, 2, 3], [1, 2, 3]]


def test_values_str(rc: RunCollection):
    assert rc.values("port") == [2, 1]


def test_values_list(rc: RunCollection):
    values = rc.values(["host", "data.y"])
    assert all(v == ("localhost", [4, 5, 6]) for v in values)


def test_sorted(rc: RunCollection):
    rc = rc.sorted("port")
    assert rc.values("port") == [1, 2]


@pytest.mark.parametrize(("port", "n"), [(1, 1), (2, 1), (3, 0)])
def test_filter_dict(rc: RunCollection, port, n):
    assert len(rc.filter({"port": port})) == n


def test_config(get_config):
    cfg: Config = get_config("run/run.py")
    assert cfg.host == "localhost"
    assert cfg.port == 3306
    assert cfg.data.x == [1, 2, 3]
