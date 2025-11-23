from __future__ import annotations

import time
from typing import TYPE_CHECKING

from joblib import Parallel, delayed  # pyright: ignore[reportUnknownVariableType]

if TYPE_CHECKING:
    import pytest


def f(x: int) -> int:
    time.sleep(0.1)
    return x


def test_progress(capsys: pytest.CaptureFixture[str]):
    from hydraflow.utils.progress import Progress

    with Progress(*Progress.get_default_columns()) as progress:
        progress.add_task("test", total=100)
        Parallel(n_jobs=8, backend="threading")(delayed(f)(i) for i in range(100))

    out = capsys.readouterr().out
    assert "test" in out
    assert "100%" in out
    assert "100/100" in out


def test_progress_without_tasks(capsys: pytest.CaptureFixture[str]):
    from hydraflow.utils.progress import Progress

    with Progress(*Progress.get_default_columns()):
        Parallel(n_jobs=8, backend="threading")(delayed(f)(i) for i in range(100))

    out = capsys.readouterr().out
    assert "100/?" in out
