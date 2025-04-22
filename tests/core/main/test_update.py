from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def results(collect):
    file = Path(__file__).parent / "update.py"
    collect(file, ["width=3", "height=4"])
    collect(file, ["width=3", "area=12"])
    collect(file, ["height=4", "area=12"])
    collect(file, ["width=5", "height=5"])
    collect(file, ["width=5", "area=25"])
    return collect(file, ["height=5", "area=35"])


def test_len(results):
    assert len(results) == 3
