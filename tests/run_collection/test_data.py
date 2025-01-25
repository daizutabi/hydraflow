import mlflow
import pytest
from mlflow.entities import Experiment

from hydraflow.run_collection import RunCollection


@pytest.fixture(scope="module")
def experiment(experiment_name: str):
    experiment = mlflow.set_experiment(experiment_name)

    for x in range(3):
        with mlflow.start_run(run_name=f"{x}"):
            mlflow.log_param("p", x)
            mlflow.log_metric("metric1", x + 1)
            mlflow.log_metric("metric2", x + 2)

    return experiment


@pytest.fixture
def rc(experiment: Experiment):
    from hydraflow.mlflow import search_runs

    return search_runs(experiment_names=[experiment.name])


def test_data_params(rc: RunCollection):
    assert rc.data.params["p"] == ["0", "1", "2"]


def test_data_metrics(rc: RunCollection):
    m = rc.data.metrics
    assert m["metric1"] == [1, 2, 3]
    assert m["metric2"] == [2, 3, 4]


def test_data_empty_run_collection():
    rc = RunCollection([])
    assert rc.data.params == {}
    assert rc.data.metrics == {}
    assert len(rc.data.config) == 0
