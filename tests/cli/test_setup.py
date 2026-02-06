from __future__ import annotations

from pathlib import Path


def test_setup() -> None:
    assert Path("hydraflow.yaml").exists()
    assert Path("app.py").exists()
    assert Path.cwd() != Path(__file__).parent
