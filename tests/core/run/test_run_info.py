from pathlib import Path

import pytest

from hydraflow.core.run_info import RunInfo

text = """\
    - cx=5e-09
    - cz=5e-09
  job:
    name: hello
    chdir: null
    id: '0'
    num: 0
"""


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory):
    p = tmp_path_factory.mktemp("artifacts", numbered=False)
    (p / ".hydra").mkdir()
    return p.parent


def test_job_name(run_dir: Path):
    run_dir.joinpath("artifacts/.hydra/hydra.yaml").write_text(text)
    assert RunInfo(run_dir).job_name == "hello"


def test_job_name_invalid_file(run_dir: Path):
    run_dir.joinpath("artifacts/.hydra/hydra.yaml").write_text("invalid")
    assert RunInfo(run_dir).job_name == ""


def test_job_name_no_file():
    assert RunInfo(Path()).job_name == ""


def test_run_id():
    assert RunInfo(Path(__file__)).run_id == "test_run_info.py"
