import pytest

from hydraflow.run_collection import RunCollection


@pytest.fixture(scope="module")
def rc(collect):
    collect("run/filter.py", ["-m", "port=1,2,3"])
    return collect("run/filter.py", ["-m", "port=1,2,4"])


def test_rc_len(rc: RunCollection):
    assert len(rc) == 4


def test_params(rc: RunCollection):
    assert rc.data.params["port"] == ["1", "2", "3", "4"]
