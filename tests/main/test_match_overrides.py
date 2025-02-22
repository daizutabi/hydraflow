import pytest

from hydraflow.entities.run_collection import RunCollection

pytestmark = pytest.mark.xdist_group(name="group5")


@pytest.fixture(scope="module")
def rc(collect):
    collect("main/match_overrides.py", ["-m", "count=1,2"])
    collect("main/match_overrides.py", ["-m", "name=a,b"])
    return collect("main/match_overrides.py", ["-m", "name=a,b"])


def test_rc_len(rc: RunCollection):
    assert len(rc) == 4


def test_config(rc: RunCollection):
    df = rc.data.config
    assert len(df) == 4
    assert len(df.drop_duplicates()) == 3
