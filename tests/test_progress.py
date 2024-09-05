import sys
from subprocess import run


def test_progress_bar():
    cp = run([sys.executable, "-m", "hydraflow.progress"])
    assert cp.returncode == 0
