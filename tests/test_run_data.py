import mlflow
import pytest

from hydraflow.run_collection import RunCollection


@pytest.fixture(scope="module")
def experiment_name(experiment_name: str):
    for x in range(3):
        with mlflow.start_run(run_name=f"{x}"):
            mlflow.log_param("p", x)
            mlflow.log_metric("metric1", x + 1)
            mlflow.log_metric("metric2", x + 2)

    yield experiment_name


@pytest.fixture
def rc(experiment_name: str):
    from hydraflow.mlflow import search_runs

    return search_runs(experiment_names=[experiment_name])


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
