from __future__ import annotations

import shutil
import subprocess
import sys
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
        ["width=1", "height=1"],
        ["-m", "width=2,3", "height=4,5", "--dry-run"],
    )


def test_len(results: Results):
    assert len(results) == 1


def test_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    file = Path(__file__).parent.joinpath("update.py")
    shutil.copy(file, "update.py")
    args = [sys.executable, "update.py", "-m", "width=2,3", "height=4,5", "--dry-run"]
    out = subprocess.check_output(args, text=True)
    assert "width: 2\nheight: 4\narea: 8" in out
    assert "width: 2\nheight: 5\narea: 10" in out
    assert "width: 3\nheight: 4\narea: 12" in out
    assert "width: 3\nheight: 5\narea: 15" in out
