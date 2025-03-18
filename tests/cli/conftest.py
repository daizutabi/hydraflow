from pathlib import Path
from shutil import copy

import pytest


@pytest.fixture(scope="module", autouse=True)
def setup(chdir):
    src = Path(__file__).parent / "hydraflow.yaml"
    copy(src, src.name)
    src = Path(__file__).parent / "app.py"
    copy(src, src.name)
    src = Path(__file__).parent / "submit.py"
    copy(src, src.name)
