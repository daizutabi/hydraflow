from __future__ import annotations

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import pytest
from mlflow.artifacts import download_artifacts
from mlflow.entities.run import Run


@pytest.fixture(scope="module")
def runs():
    file = Path("tests/log_run.py").absolute()
    curdir = Path.cwd()

    with TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)

        subprocess.check_call(["python", file, "-m", "host=x,y", "port=1,2"])

        mlflow.set_experiment("log_run")
        runs = mlflow.search_runs(output_format="list")
        assert len(runs) == 4
        assert isinstance(runs, list)
        yield runs

    os.chdir(curdir)


@pytest.fixture(params=range(4))
def run_id(runs, request):
    run = runs[request.param]
    assert isinstance(run, Run)
    return run.info.run_id


def test_output(run_id: str):
    path = download_artifacts(run_id=run_id, artifact_path="a.txt")
    text = Path(path).read_text()
    assert text == "abc"


def read_log(run_id: str) -> str:
    path = download_artifacts(run_id=run_id, artifact_path="log_run.log")
    text = Path(path).read_text()
    assert "START" in text
    assert "END" in text
    return text


def test_load_config(run_id: str):
    from hydraflow.run import load_config

    log = read_log(run_id)
    host, port = log.splitlines()[0].split("START,")[-1].split(",")

    cfg = load_config(run_id)
    assert cfg.host == host.strip()
    assert cfg.port == int(port)


def test_load_config_err(run_id: str):
    from hydraflow.run import load_config

    assert not load_config(run_id, "a")
