from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import pytest
from mlflow.entities import RunStatus
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from hydraflow.run_collection import RunCollection


@pytest.fixture(scope="module")
def rc(tmp_path_factory: pytest.TempPathFactory):
    import hydraflow

    cwd = Path.cwd()

    file = Path("tests/scripts/app.py").absolute()
    os.chdir(tmp_path_factory.mktemp("test_app"))

    args = [sys.executable, file.as_posix(), "-m"]
    args += ["host=x,y", "port=1,2", "hydra.job.name=info"]
    subprocess.check_call(args)

    mlflow.set_experiment("_info_")
    yield hydraflow.list_runs()

    os.chdir(cwd)


def test_list_runs_all(rc: RunCollection):
    from hydraflow.mlflow import list_runs

    rc_ = list_runs([])
    assert len(rc) == len(rc_)

    for a, b in zip(rc, rc_, strict=False):
        assert a.info.run_id == b.info.run_id
        assert a.info.start_time == b.info.start_time
        assert a.info.status == b.info.status
        assert a.info.artifact_uri == b.info.artifact_uri


def test_list_runs_status(rc: RunCollection):
    from hydraflow.mlflow import list_runs

    rc_ = list_runs("_info_", status="finished")
    assert len(rc) == len(rc_)
    rc_ = list_runs("_info_", status=RunStatus.FINISHED)
    assert len(rc) == len(rc_)
    assert not list_runs("_info_", status=RunStatus.RUNNING)
    rc_ = list_runs("_info_", status="!RUNNING")
    assert len(rc) == len(rc_)
    assert not list_runs("_info_", status="!FINISHED")


@pytest.mark.parametrize("n_jobs", [0, 1, 2, 4, -1])
def test_list_runs_parallel(rc: RunCollection, n_jobs: int):
    from hydraflow.mlflow import list_runs

    rc_ = list_runs("_info_", n_jobs=n_jobs)
    assert len(rc) == len(rc_)

    for a, b in zip(rc, rc_, strict=False):
        assert a.info.run_id == b.info.run_id
        assert a.info.start_time == b.info.start_time
        assert a.info.status == b.info.status
        assert a.info.artifact_uri == b.info.artifact_uri


@pytest.mark.parametrize("n_jobs", [0, 1, 2, 4, -1])
def test_list_runs_parallel_active(rc: RunCollection, n_jobs: int):
    from hydraflow.mlflow import list_runs

    mlflow.set_experiment("_info_")
    rc_ = list_runs(n_jobs=n_jobs)
    assert len(rc) == len(rc_)

    for a, b in zip(rc, rc_, strict=False):
        assert a.info.run_id == b.info.run_id
        assert a.info.start_time == b.info.start_time
        assert a.info.status == b.info.status
        assert a.info.artifact_uri == b.info.artifact_uri


def test_app_info_run_id(rc: RunCollection):
    assert len(rc.info.run_id) == 4


def test_app_data_params(rc: RunCollection):
    params = rc.data.params
    assert params["port"] == ["1", "2", "1", "2"]
    assert params["host"] == ["x", "x", "y", "y"]
    assert params["values"] == ["[1, 2, 3]", "[1, 2, 3]", "[1, 2, 3]", "[1, 2, 3]"]


def test_app_data_metrics(rc: RunCollection):
    metrics = rc.data.metrics
    assert metrics["m"] == [11, 12, 2, 3]
    # assert metrics["watch"] == [3, 3, 3, 3]  # noqa: ERA001


def test_app_data_config(rc: RunCollection):
    config = rc.data.config
    assert config["port"].to_list() == [1, 2, 1, 2]
    assert config["host"].to_list() == ["x", "x", "y", "y"]


def test_app_data_config_list(rc: RunCollection):
    config = rc.data.config
    values = config["values"].to_list()
    assert str(config.select("values").dtypes) == "[List(Int64)]"
    for x in values:
        assert isinstance(x, list)
        assert x == [1, 2, 3]


def test_app_info_artifact_uri(rc: RunCollection):
    uris = rc.info.artifact_uri
    assert all(uri.startswith("file://") for uri in uris)  # type: ignore
    assert all(uri.endswith("/artifacts") for uri in uris)  # type: ignore
    assert all("mlruns" in uri for uri in uris)  # type: ignore


def test_app_info_artifact_dir(rc: RunCollection):
    from hydraflow.utils import get_artifact_dir

    dirs = list(rc.map(get_artifact_dir))
    assert rc.info.artifact_dir == dirs


def test_app_hydra_output_dir(rc: RunCollection):
    from hydraflow.utils import get_hydra_output_dir

    dirs = list(rc.map(get_hydra_output_dir))
    assert dirs[0].stem == "0"
    assert dirs[1].stem == "1"
    assert dirs[2].stem == "2"
    assert dirs[3].stem == "3"


def test_app_map_config(rc: RunCollection):
    ports = []

    def func(c: DictConfig, *, a: int):
        ports.append(c.port + 1)
        return c.host

    hosts = list(rc.map_config(func, a=1))
    assert hosts == ["x", "x", "y", "y"]
    assert ports == [2, 3, 2, 3]


def test_app_group_by(rc: RunCollection):
    grouped = rc.group_by("host")
    assert len(grouped) == 2
    assert grouped["x"].data.params["port"] == ["1", "2"]
    assert grouped["x"].data.params["host"] == ["x", "x"]
    assert grouped["x"].data.params["values"] == ["[1, 2, 3]", "[1, 2, 3]"]
    assert grouped["y"].data.params["port"] == ["1", "2"]
    assert grouped["y"].data.params["host"] == ["y", "y"]
    assert grouped["y"].data.params["values"] == ["[1, 2, 3]", "[1, 2, 3]"]


def test_app_group_by_list(rc: RunCollection):
    grouped = rc.group_by(["host"])
    assert len(grouped) == 2
    assert ("x",) in grouped
    assert ("y",) in grouped


def test_app_filter_list(rc: RunCollection):
    filtered = rc.filter(values=[1, 2, 3])
    assert len(filtered) == 4
    filtered = rc.filter(values=OmegaConf.create([1, 2, 3]))
    assert len(filtered) == 4
    filtered = rc.filter(values=[1])
    assert not filtered


def test_values(rc: RunCollection):
    values = rc.values("host")
    assert values == ["x", "x", "y", "y"]
    values = rc.values(["host", "port"])
    assert values == [("x", 1), ("x", 2), ("y", 1), ("y", 2)]


def test_sort_by(rc: RunCollection):
    sorted = rc.sort_by("host", reverse=True)
    assert sorted.values(["host", "port"]) == [("y", 1), ("y", 2), ("x", 1), ("x", 2)]

    sorted = rc.sort_by(["host", "port"], reverse=True)
    assert sorted.values(["host", "port"]) == [("y", 2), ("y", 1), ("x", 2), ("x", 1)]


def test_log_run_error(monkeypatch, tmp_path):
    file = Path("tests/scripts/app.py").absolute()
    monkeypatch.chdir(tmp_path)

    args = [sys.executable, file.as_posix()]
    args += ["host=error", "hydra.job.name=error"]
    cp = subprocess.run(args, check=False, capture_output=True)
    assert cp.returncode


def test_chdir_artifact(rc: RunCollection):
    from hydraflow.context import chdir_artifact

    with chdir_artifact(rc[0]):
        assert Path.cwd().stem == "artifacts"
        assert Path.cwd().parent.stem == rc[0].info.run_id
