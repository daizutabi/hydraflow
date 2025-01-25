import mlflow
import pytest
from mlflow.entities import Experiment

from hydraflow.run_collection import RunCollection


@pytest.fixture(scope="module")
def experiment(experiment_name: str):
    experiment = mlflow.set_experiment(experiment_name)

    for x in range(4):
        with mlflow.start_run(run_name=f"{x}"):
            pass

    return experiment


@pytest.fixture
def rc(experiment: Experiment):
    from hydraflow.mlflow import search_runs

    return search_runs(experiment_names=[experiment.name])


def test_remove_run(rc: RunCollection):
    from hydraflow.utils import get_artifact_dir, remove_run

    paths = [get_artifact_dir(r).parent for r in rc]

    assert all(path.exists() for path in paths)

    remove_run(rc)

    assert not any(path.exists() for path in paths)
