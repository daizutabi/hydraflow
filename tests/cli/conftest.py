from __future__ import annotations

import os
from pathlib import Path
from shutil import copy

import pytest


@pytest.fixture(autouse=True)
def setup(tmp_path: Path):
    orig_dir = Path().absolute()

    os.chdir(tmp_path)

    src = Path(__file__).parent / "hydraflow.yaml"
    copy(src, src.name)
    src = Path(__file__).parent / "app.py"
    copy(src, src.name)
    src = Path(__file__).parent / "submit.py"
    copy(src, src.name)

    try:
        yield
    finally:
        os.chdir(orig_dir)
