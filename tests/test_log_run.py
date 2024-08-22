from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mlflow
import pytest
from mlflow.artifacts import download_artifacts
from mlflow.entities.run import Run


@pytest.fixture
def runs(monkeypatch, tmp_path):
    file = Path("tests/scripts/log_run.py").absolute()
    monkeypatch.chdir(tmp_path)

    subprocess.check_call([sys.executable, file.as_posix(), "-m", "host=x,y", "port=1,2"])

    mlflow.set_experiment("log_run")
    runs = mlflow.search_runs(output_format="list")
    assert len(runs) == 4
    assert isinstance(runs, list)
    yield runs


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
    from hydraflow.runs import load_config

    log = read_log(run_id)
    host, port = log.splitlines()[0].split("START,")[-1].split(",")

    cfg = load_config(run_id)
    assert cfg.host == host.strip()
    assert cfg.port == int(port)
