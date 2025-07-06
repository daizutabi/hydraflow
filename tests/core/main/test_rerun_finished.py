from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def results(collect):
    file = Path(__file__).parent / "rerun_finished.py"
    results = None
    for _ in range(3):
        results = collect(file, ["count=3"])
    return results


def test_len(results):
    assert len(results) == 1


def test_count(results):
    path: Path = results[0][0]
    assert path.joinpath("a.txt").read_text() == "333"
