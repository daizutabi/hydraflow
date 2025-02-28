import sys
from pathlib import Path

import pytest
from mlflow.entities import Run

from hydraflow.entities.run_collection import RunCollection

pytestmark = pytest.mark.xdist_group(name="group6")


@pytest.fixture(scope="module")
def rc(collect):
    file = Path(__file__).parent / "hydra_dir.py"
    return collect(file, ["-m", "name=a,b", "age=10"])


def test_rc_len(rc: RunCollection):
    assert len(rc) == 2


@pytest.fixture(scope="module")
def run(rc: RunCollection):
    return rc.first()


@pytest.mark.parametrize(
    ("uri", "path"),
    [("/a/b/c", "/a/b/c"), ("file:///a/b/c", "/a/b/c"), ("file:C:/a/b/c", "C:/a/b/c")],
)
def test_file_uri_to_path(uri, path):
    from hydraflow.core.io import file_uri_to_path

    assert file_uri_to_path(uri).as_posix() == path


@pytest.mark.skipif(sys.platform != "win32", reason="This test is for Windows")
def test_file_uri_to_path_win_python_310_311():
    from hydraflow.core.io import file_uri_to_path

    assert file_uri_to_path("file:///C:/a/b/c").as_posix() == "C:/a/b/c"


def test_hydra_output_dir(run: Run):
    from hydraflow.core.io import get_artifact_path, get_hydra_output_dir

    path = get_artifact_path(run, "hydra_output_dir.txt")
    assert get_hydra_output_dir(run).as_posix() == path.read_text()


def test_load_config(run: Run):
    from hydraflow.core.io import load_config

    cfg = load_config(run)
    assert cfg.name == "a"
    assert cfg.age == 10
    assert cfg.height == 1.7


def test_load_overrides(run: Run):
    from hydraflow.core.io import load_overrides

    overrides = load_overrides(run)
    assert overrides == ["age=10", "name=a"]
