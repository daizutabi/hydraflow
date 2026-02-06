from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.core.conftest import Collect, Results


@pytest.fixture(scope="module")
def results(collect: Collect) -> Results:
    file = Path(__file__).parent / "update.py"
    return collect(
        file,
        ["width=3", "height=4"],
        ["width=3", "area=12"],
        ["height=4", "area=12"],
        ["width=5", "height=5"],
        ["width=5", "area=25"],
        ["height=5", "area=35"],
    )


def test_len(results: Results) -> None:
    assert len(results) == 3
