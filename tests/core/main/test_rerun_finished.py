from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.core.conftest import Collect, Results


@pytest.fixture(scope="module")
def results(collect: Collect) -> Results:
    file = Path(__file__).parent / "rerun_finished.py"
    return collect(file, ["count=3"], ["count=3"], ["count=3"])


def test_len(results: Results) -> None:
    assert len(results) == 1


def test_count(results: Results) -> None:
    path = results[0][0]
    assert path.joinpath("a.txt").read_text() == "333"
