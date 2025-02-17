from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    yield
