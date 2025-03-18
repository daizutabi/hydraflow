import sys

import pytest


@pytest.mark.skipif(sys.platform == "win32", reason="Not supported on Windows")
def test_run_returncode():
    from hydraflow.executor.aio import run
    from hydraflow.executor.job import Task

    task = Task(args=["false"], index=0, total=1)

    assert run([task]) == 1


def test_run_stderr():
    from hydraflow.executor.aio import run
    from hydraflow.executor.job import Task

    task = Task(args=["python", "-c", "1/0"], index=0, total=1)

    assert run([task]) == 1
