import sys
from typing import TYPE_CHECKING

import pytest
from mlflow.entities import Run

from hydraflow.run_collection import RunCollection

if TYPE_CHECKING:
    from .utils import Config

pytestmark = pytest.mark.xdist_group(name="group6")


@pytest.fixture(scope="module")
def rc(collect):
    return collect("utils/utils.py", ["-m", "name=a,b", "age=10"])


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
    from hydraflow.utils import file_uri_to_path

    assert file_uri_to_path(uri).as_posix() == path


@pytest.mark.skipif(sys.platform != "win32", reason="This test is for Windows")
def test_file_uri_to_path_win_python_310_311():
    from hydraflow.utils import file_uri_to_path

    assert file_uri_to_path("file:///C:/a/b/c").as_posix() == "C:/a/b/c"


def test_hydra_output_dir(run: Run):
    from hydraflow.utils import get_artifact_path, get_hydra_output_dir

    path = get_artifact_path(run, "hydra_output_dir.txt")
    assert get_hydra_output_dir(run).as_posix() == path.read_text()


def test_load_config(run: Run):
    from hydraflow.utils import load_config

    cfg: Config = load_config(run)  # type: ignore
    assert cfg.name == "a"
    assert cfg.age == 10
    assert cfg.height == 1.7


def test_load_overrides(run: Run):
    from hydraflow.utils import load_overrides

    overrides = load_overrides(run)
    assert overrides == ["name=a", "age=10"]
