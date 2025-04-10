from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def results(collect):
    file = Path(__file__).parent / "match_overrides.py"
    collect(file, ["-m", "count=1,2"])
    collect(file, ["-m", "name=a,b", "count=1"])
    return collect(file, ["-m", "count=1", "name=a,b"])


def test_len(results):
    assert len(results) == 4
