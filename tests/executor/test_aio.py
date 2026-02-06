from __future__ import annotations

import sys

import pytest

from hydraflow.executor.aio import run
from hydraflow.executor.job import Task


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on Windows")
def test_run_returncode() -> None:
    task = Task(args=["false"], index=0, total=1)
    assert run([task]) == 1


def test_run_stderr() -> None:
    task = Task(args=["python", "-c", "1/0"], index=0, total=1)
    assert run([task]) == 1
