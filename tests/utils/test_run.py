import mlflow
import pytest
from mlflow.entities import Experiment, Run

from hydraflow.run_collection import RunCollection


@pytest.fixture(scope="module")
def experiment(experiment_name: str):
    experiment = mlflow.set_experiment(experiment_name)

    for x in range(4):
        with mlflow.start_run(run_name=f"{x}"):
            pass

    return experiment


@pytest.fixture(scope="module")
def rc(experiment: Experiment):
    from hydraflow.mlflow import search_runs

    return search_runs(experiment_names=[experiment.name])


@pytest.fixture(scope="module")
def run(rc: RunCollection):
    return rc.first()


def test_hydra_output_dir(run: Run):
    from hydraflow.utils import get_hydra_output_dir

    with pytest.raises(FileNotFoundError):
        get_hydra_output_dir(run)


def test_remove_run(rc: RunCollection):
    from hydraflow.utils import get_artifact_dir, remove_run

    paths = [get_artifact_dir(r).parent for r in rc]

    assert all(path.exists() for path in paths)

    remove_run(rc)

    assert not any(path.exists() for path in paths)
