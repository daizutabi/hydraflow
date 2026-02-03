from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import pytest

from hydraflow.core.io import (
    file_uri_to_path,
    get_artifact_dir,
    get_experiment_name,
    get_experiment_names,
    iter_artifact_paths,
    iter_artifacts_dirs,
    iter_experiment_dirs,
    iter_run_dirs,
    log_text,
)

if TYPE_CHECKING:
    from mlflow.entities import Run


@pytest.mark.parametrize(
    ("uri", "path"),
    [("/a/b/c", "/a/b/c"), ("file:///a/b/c", "/a/b/c"), ("file:C:/a/b/c", "C:/a/b/c")],
)
def test_file_uri_to_path(uri: str, path: str):
    assert file_uri_to_path(uri).as_posix() == path


@pytest.mark.skipif(sys.platform != "win32", reason="This test is for Windows")
def test_file_uri_to_path_win_python_310_311():
    assert file_uri_to_path("file:///C:/a/b/c").as_posix() == "C:/a/b/c"


@pytest.fixture(scope="module", autouse=True, params=["mlruns", "mlflow.db"])
def setup(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
):
    curdir = Path.cwd()
    tmpdir = tmp_path_factory.mktemp(request.param)
    os.chdir(tmpdir)
    assert isinstance(request.param, str)

    if "." in request.param:
        db = (tmpdir / request.param).as_posix()
        uri = f"sqlite:///{db}"
    else:
        uri = tmpdir / request.param

    Path("123.log").write_text("test log", encoding="utf-8")
    Path("dir.log").mkdir()

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("e1")
    with mlflow.start_run() as run:
        mlflow.log_text("1", "text.txt")
        log_text(run, Path.cwd(), "*.log")
        log_text(run, Path.cwd(), "*.log")

    with mlflow.start_run():
        mlflow.log_text("2", "text.txt")

    mlflow.set_experiment("e2")
    with mlflow.start_run():
        mlflow.log_text("3", "text.txt")
    with mlflow.start_run():
        mlflow.log_text("4", "text.txt")
    with mlflow.start_run():
        mlflow.log_text("5", "text.txt")

    yield

    os.chdir(curdir)


@pytest.fixture(scope="module")
def run() -> Run:
    runs = mlflow.search_runs(experiment_names=["e1"], output_format="list")
    assert isinstance(runs, list)
    return runs[-1]


def test_get_artifact_dir(run: Run) -> None:
    artifact_dir = get_artifact_dir(run)
    assert artifact_dir.name == "artifacts"
    assert artifact_dir.parent.name == run.info.run_id


def test_log_text(run: Run) -> None:
    artifact_dir = get_artifact_dir(run)
    logged_file = artifact_dir / "123.log"
    assert logged_file.exists()
    assert logged_file.read_text().endswith("test log\ntest log")


def test_get_experiment_names():
    assert sorted(get_experiment_names()) == ["e1", "e2"]


def test_iter_experiment_dirs():
    names = [get_experiment_name(p) for p in iter_experiment_dirs()]
    assert sorted(names) == ["e1", "e2"]


@pytest.mark.parametrize(
    ("e", "es"),
    [("e1", ["e1"]), ("e*", ["e1", "e2"]), ("*", ["e1", "e2"]), ("*2", ["e2"])],
)
def test_iter_experiment_dirs_glob(e: str, es: list[str]):
    names = [get_experiment_name(p) for p in iter_experiment_dirs(e)]
    assert sorted(names) == es


def test_iter_experiment_dirs_filter():
    it = iter_experiment_dirs(experiment_names="e1")
    assert [get_experiment_name(p) for p in it] == ["e1"]


def test_iter_experiment_dirs_filter_callable():
    it = iter_experiment_dirs(experiment_names=lambda name: name == "e2")
    assert [get_experiment_name(p) for p in it] == ["e2"]


def test_get_experiment_name_none():
    assert get_experiment_name("invalid") == ""


def test_iter_run_dirs():
    assert len(list(iter_run_dirs())) == 5


def test_iter_artifacts_dirs():
    assert len(list(iter_artifacts_dirs())) == 5


def test_iter_artifact_paths():
    it = iter_artifact_paths("text.txt")
    text = sorted("".join(p.read_text() for p in it))
    assert text == ["1", "2", "3", "4", "5"]
