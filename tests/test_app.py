from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import pytest
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from hydraflow.run_collection import RunCollection


@pytest.fixture
def rc(monkeypatch, tmp_path):
    import hydraflow

    file = Path("tests/scripts/app.py").absolute()
    monkeypatch.chdir(tmp_path)

    args = [sys.executable, file.as_posix(), "-m"]
    args += ["host=x,y", "port=1,2", "hydra.job.name=info"]
    subprocess.check_call(args)

    mlflow.set_experiment("_info_")
    yield hydraflow.list_runs()


def test_list_runs_all(rc: RunCollection):
    from hydraflow.mlflow import list_runs

    rc_ = list_runs([])
    assert len(rc) == len(rc_)

    for a, b in zip(rc, rc_, strict=False):
        assert a.info.run_id == b.info.run_id
        assert a.info.start_time == b.info.start_time
        assert a.info.status == b.info.status
        assert a.info.artifact_uri == b.info.artifact_uri


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


def test_app_info_params(rc: RunCollection):
    params = rc.info.params
    assert params[0] == {"port": "1", "host": "x", "values": "[1, 2, 3]"}
    assert params[1] == {"port": "2", "host": "x", "values": "[1, 2, 3]"}
    assert params[2] == {"port": "1", "host": "y", "values": "[1, 2, 3]"}
    assert params[3] == {"port": "2", "host": "y", "values": "[1, 2, 3]"}


def test_app_info_metrics(rc: RunCollection):
    metrics = rc.info.metrics
    assert metrics[0] == {"m": 11, "watch": 3}
    assert metrics[1] == {"m": 12, "watch": 3}
    assert metrics[2] == {"m": 2, "watch": 3}
    assert metrics[3] == {"m": 3, "watch": 3}


def test_app_info_config(rc: RunCollection):
    config = rc.info.config
    assert config[0].port == 1
    assert config[1].port == 2
    assert config[2].host == "y"
    assert config[3].host == "y"


def test_app_info_artifact_uri(rc: RunCollection):
    uris = rc.info.artifact_uri
    assert all(uri.startswith("file://") for uri in uris)  # type: ignore
    assert all(uri.endswith("/artifacts") for uri in uris)  # type: ignore
    assert all("mlruns" in uri for uri in uris)  # type: ignore


def test_app_info_artifact_dir(rc: RunCollection):
    from hydraflow.info import get_artifact_dir

    dirs = list(rc.map(get_artifact_dir))
    assert rc.info.artifact_dir == dirs


def test_app_hydra_output_dir(rc: RunCollection):
    from hydraflow.info import get_hydra_output_dir

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
    x = {"port": "1", "host": "x", "values": "[1, 2, 3]"}
    assert grouped[("x",)].info.params[0] == x
    x = {"port": "2", "host": "x", "values": "[1, 2, 3]"}
    assert grouped[("x",)].info.params[1] == x
    x = {"port": "1", "host": "y", "values": "[1, 2, 3]"}
    assert grouped[("y",)].info.params[0] == x
    x = {"port": "2", "host": "y", "values": "[1, 2, 3]"}
    assert grouped[("y",)].info.params[1] == x


def test_app_filter_list(rc: RunCollection):
    filtered = rc.filter(values=[1, 2, 3])
    assert len(filtered) == 4
    filtered = rc.filter(values=OmegaConf.create([1, 2, 3]))
    assert len(filtered) == 4
    filtered = rc.filter(values=[1])
    assert not filtered
