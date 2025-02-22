import mlflow
import pytest
from mlflow.entities import Experiment, Run

from hydraflow.entities.run_collection import RunCollection

pytestmark = pytest.mark.xdist_group(name="group5")


@pytest.fixture(scope="module")
def experiment(experiment_name: str):
    experiment = mlflow.set_experiment(experiment_name)

    for x in range(4):
        with mlflow.start_run(run_name=f"{x}"):
            pass

    return experiment


@pytest.fixture(scope="module")
def rc(experiment: Experiment):
    from hydraflow.core.mlflow import list_runs

    return list_runs(experiment.name)


@pytest.fixture(scope="module")
def run(rc: RunCollection):
    return rc.first()


@pytest.mark.order(0)
def test_hydra_output_dir(run: Run):
    from hydraflow.core.io import get_hydra_output_dir

    with pytest.raises(FileNotFoundError):
        get_hydra_output_dir(run)


@pytest.mark.order(1)
def test_remove_run(rc: RunCollection):
    from hydraflow.core.io import get_artifact_dir, remove_run

    paths = [get_artifact_dir(r).parent for r in rc]

    assert all(path.exists() for path in paths)

    remove_run(rc)

    assert not any(path.exists() for path in paths)
