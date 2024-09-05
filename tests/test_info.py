from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from hydraflow.run_collection import RunCollection


@pytest.fixture
def runs(monkeypatch, tmp_path):
    from hydraflow.run_collection import search_runs

    monkeypatch.chdir(tmp_path)

    mlflow.set_experiment("test_info")

    for x in range(3):
        with mlflow.start_run(run_name=f"{x}"):
            mlflow.log_param("p", x)
            mlflow.log_metric("metric1", x + 1)
            mlflow.log_metric("metric2", x + 2)

    x = search_runs()
    assert isinstance(x, RunCollection)
    return x


def test_info_run_id(runs: RunCollection):
    assert len(runs.info.run_id) == 3


def test_info_params(runs: RunCollection):
    assert runs.info.params == [{"p": "0"}, {"p": "1"}, {"p": "2"}]


def test_info_metrics(runs: RunCollection):
    m = runs.info.metrics
    assert m[0] == {"metric1": 1, "metric2": 2}
    assert m[1] == {"metric1": 2, "metric2": 3}
    assert m[2] == {"metric1": 3, "metric2": 4}


def test_info_artifact_uri(runs: RunCollection):
    uri = runs.info.artifact_uri
    assert all(u.startswith("file://") for u in uri)  # type: ignore
    assert all(u.endswith("/artifacts") for u in uri)  # type: ignore


def test_info_artifact_dir(runs: RunCollection):
    dir = runs.info.artifact_dir
    assert all(isinstance(d, Path) for d in dir)
    assert all(d.stem == "artifacts" for d in dir)  # type: ignore


def test_info_empty_run_collection():
    rc = RunCollection([])
    assert rc.info.run_id == []
    assert rc.info.params == []
    assert rc.info.metrics == []
    assert rc.info.artifact_uri == []
    assert rc.info.artifact_dir == []
    assert rc.info.config == []
