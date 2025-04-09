from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def results(collect):
    file = Path(__file__).parent / "force_new_run.py"
    for _ in range(3):
        results = collect(file, ["count=3"])
    return results


def test_len(results):
    assert len(results) == 3


@pytest.fixture(scope="module", params=range(3))
def result(results, request: pytest.FixtureRequest):
    return results[request.param]


@pytest.fixture(scope="module")
def path(result):
    return result[0]


def test_count(path: Path):
    assert path.joinpath("a.txt").read_text() == "3"
