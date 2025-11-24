from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from tests.core.conftest import Collect, Results


@pytest.fixture(scope="module")
def results(collect: Collect) -> Results:
    file = Path(__file__).parent / "match_overrides.py"
    return collect(
        file,
        ["-m", "count=1,2"],
        ["-m", "name=a,b", "count=1"],
        ["-m", "count=1", "name=a,b"],
    )


def test_len(results: Results):
    assert len(results) == 4
