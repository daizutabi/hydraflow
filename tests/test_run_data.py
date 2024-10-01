from __future__ import annotations

import mlflow
import pytest

from hydraflow.run_collection import RunCollection


@pytest.fixture
def runs(monkeypatch, tmp_path):
    from hydraflow.mlflow import search_runs

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


def test_data_params(runs: RunCollection):
    assert runs.data.params["p"] == ["0", "1", "2"]


def test_data_metrics(runs: RunCollection):
    m = runs.data.metrics
    assert m["metric1"] == [1, 2, 3]
    assert m["metric2"] == [2, 3, 4]


def test_data_empty_run_collection():
    rc = RunCollection([])
    assert rc.data.params == {}
    assert rc.data.metrics == {}
    assert len(rc.data.config) == 0
