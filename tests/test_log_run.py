from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mlflow
import pytest
from mlflow.artifacts import download_artifacts
from mlflow.entities.run import Run


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory):
    file = Path("tests/apps/app.py").absolute()

    cwd = Path.cwd()
    os.chdir(tmp_path_factory.mktemp("test_log_run"))

    args = [sys.executable, file.as_posix(), "-m"]
    args += ["host=x,y", "port=1,2", "hydra.job.name=log_run"]
    subprocess.check_call(args)

    mlflow.set_experiment("_log_run_")
    runs = mlflow.search_runs(output_format="list")

    assert len(runs) == 4
    assert isinstance(runs, list)
    yield runs

    os.chdir(cwd)


@pytest.fixture(params=range(4))
def run(runs, request):
    run = runs[request.param]  # type: ignore
    assert isinstance(run, Run)
    return run


@pytest.fixture
def run_id(run: Run):
    return run.info.run_id


def test_output(run_id: str):
    path = download_artifacts(run_id=run_id, artifact_path="a.txt")
    text = Path(path).read_text()
    assert text == "abc"


def read_log(run_id: str, path: str) -> str:
    path = download_artifacts(run_id=run_id, artifact_path=path)
    return Path(path).read_text()


def test_load_config(run: Run):
    from hydraflow.utils import load_config

    log = read_log(run.info.run_id, "log_run.log")
    assert "START" in log
    assert "END" in log

    host, port = log.splitlines()[0].split("START,")[-1].split(",")

    cfg = load_config(run)
    assert cfg.host == host.strip()
    assert cfg.port == int(port)


def test_load_overrides(run: Run):
    from hydraflow.utils import load_overrides

    log = read_log(run.info.run_id, "log_run.log")
    assert "START" in log
    assert "END" in log

    host, port = log.splitlines()[0].split("START,")[-1].split(",")

    assert load_overrides(run) == [f"host={host.strip()}", f"port={port.strip()}"]


def test_info(run: Run):
    log = read_log(run.info.run_id, "artifact_dir.txt")
    a, b = log.split(" ")
    assert a == "A"
    assert b in run.info.artifact_uri  # type: ignore

    log = read_log(run.info.run_id, "output_dir.txt")
    a, b = log.split(" ")
    assert a == "B"
    assert "multirun" in b
