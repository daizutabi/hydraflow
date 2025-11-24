from __future__ import annotations

import os
from pathlib import Path
from shutil import copy

import mlflow
import pytest


@pytest.fixture(autouse=True)
def setup(tmp_path: Path):
    orig_dir = Path().absolute()

    os.chdir(tmp_path)

    parent = Path(__file__).parent

    src = parent / "hydraflow.yaml"
    copy(src, src.name)
    src = parent / "app.py"
    copy(src, src.name)
    src = parent / "submit.py"
    copy(src, src.name)

    db = str(Path("mlflow.db").absolute())  # must be an absolute path
    mlflow.set_tracking_uri(f"sqlite:///{db}")

    try:
        yield
    finally:
        os.chdir(orig_dir)
