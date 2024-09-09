import sys
from subprocess import run

import pytest


@pytest.mark.skipif(
    sys.platform == "win32", reason="'cp932' codec can't encode character '\\u2807'"
)
def test_progress_bar():
    cp = run([sys.executable, "tests/scripts/progress.py"])
    assert cp.returncode == 0
