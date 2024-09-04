from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize("dir", [".", Path])
def test_watch(dir, monkeypatch, tmp_path):
    from hydraflow.context import watch

    file = Path("tests/scripts/watch.py").absolute()
    monkeypatch.chdir(tmp_path)

    results = []

    def func(path: Path) -> None:
        text = path.read_text()
        results.append([path.name, text])

    with watch(func, dir if isinstance(dir, str) else dir()):
        subprocess.check_call(["python", file])
        time.sleep(1)

    assert results[0][0] == "watch.txt"  # type: ignore
    assert results[0][1] == "watch"  # type: ignore
