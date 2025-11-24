from __future__ import annotations

from pathlib import Path

import pytest

from hydraflow.core.run_info import RunInfo


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory):
    p = tmp_path_factory.mktemp("artifacts", numbered=False)
    (p / ".hydra").mkdir()
    return p.parent


def test_run_id():
    assert RunInfo(Path(__file__)).run_id == "test_run_info.py"
