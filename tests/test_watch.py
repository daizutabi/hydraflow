from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize("dir", [".", Path])
def test_watch(dir, monkeypatch, tmp_path):
    from hydraflow.context import watch

    file = Path("tests/watch.py").absolute()
    monkeypatch.chdir(tmp_path)

    lines = []

    def func(path: Path) -> None:
        k, t = path.read_text().split(" ")
        lines.append([int(k), float(t), time.time(), path.name])

    with watch(func, dir if isinstance(dir, str) else dir()):
        subprocess.check_call(["python", file])

    for k in range(4):
        assert lines[k][0] == k
        assert lines[k][-1] == f"{k}.txt"
        assert 0 <= lines[k][2] - lines[k][1] < 0.05
        if k < 3:
            assert 0.95 < lines[k + 1][1] - lines[k][1] < 1.05
            assert 0.95 < lines[k + 1][2] - lines[k][2] < 1.05
