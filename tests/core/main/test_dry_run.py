import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def results(collect):
    file = Path(__file__).parent / "update.py"
    collect(file, ["width=1", "height=1"])
    return collect(file, ["-m", "width=2,3", "height=4,5", "--dry-run"])


def test_len(results):
    assert len(results) == 1


def test_log():
    file = Path(__file__).parent.joinpath("update.py").as_posix()
    args = [sys.executable, file, "-m", "width=2,3", "height=4,5", "--dry-run"]
    out = subprocess.check_output(args, text=True)
    assert "width: 2\nheight: 4\narea: 8" in out
    assert "width: 2\nheight: 5\narea: 10" in out
    assert "width: 3\nheight: 4\narea: 12" in out
    assert "width: 3\nheight: 5\narea: 15" in out
