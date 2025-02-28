from pathlib import Path

import pytest

from hydraflow.entities.run_collection import RunCollection

pytestmark = pytest.mark.xdist_group(name="group5")


@pytest.fixture(scope="module")
def rc(collect):
    file = Path(__file__).parent / "match_overrides.py"
    collect(file, ["-m", "count=1,2"])
    collect(file, ["-m", "name=a,b", "count=1"])
    return collect(file, ["-m", "count=1", "name=a,b"])


def test_rc_len(rc: RunCollection):
    assert len(rc) == 4


def test_config(rc: RunCollection):
    df = rc.data.config
    assert len(df) == 4
    assert len(df.drop_duplicates()) == 3


def test_equals():
    from hydraflow.core.main import equals

    assert equals(Path.cwd(), None, []) is False
