from pathlib import Path

import mlflow
import pytest
from mlflow.entities import Experiment, Run, RunStatus

pytestmark = pytest.mark.xdist_group(name="group0")


@pytest.fixture(scope="module")
def experiment(experiment_name: str):
    from hydraflow.core.mlflow import log_params

    mlflow.set_tracking_uri("test_mlflow")
    experiment = mlflow.set_experiment("e")

    with mlflow.start_run():
        log_params({"name": experiment_name})

    mlflow.start_run()
    mlflow.end_run(status=RunStatus.to_string(RunStatus.RUNNING))

    mlflow.start_run()
    mlflow.end_run(status=RunStatus.to_string(RunStatus.FAILED))

    return experiment


def test_set_experiment_uri(experiment: Experiment):
    assert mlflow.get_tracking_uri() == "test_mlflow"


def test_set_experiment_location(experiment: Experiment):
    loc = experiment.artifact_location
    assert isinstance(loc, str)
    if loc.startswith("file:"):  # for windows
        loc = loc[loc.index("C:") :]

    path = Path.cwd() / "test_mlflow" / experiment.experiment_id
    assert path == Path(loc)


def test_set_experiment_name(experiment: Experiment):
    e = mlflow.get_experiment_by_name("e")
    assert e
    assert e.experiment_id == experiment.experiment_id


def test_search_runs(experiment: Experiment):
    from hydraflow.core.mlflow import list_runs

    rc = list_runs(experiment.name)
    assert len(rc) == 3


@pytest.fixture(scope="module")
def run(experiment: Experiment):
    from hydraflow.core.mlflow import list_runs

    rc = list_runs(experiment.name)
    return rc.first()


def test_log_params(run: Run, experiment_name):
    assert run.data.params["name"] == experiment_name


def test_get_artifact_dir_from_utils(run: Run, experiment: Experiment):
    from hydraflow.core.io import get_artifact_dir

    loc = experiment.artifact_location
    assert isinstance(loc, str)
    if loc.startswith("file:"):  # for windows
        loc = loc[loc.index("C:") :]

    assert get_artifact_dir(run) == Path(loc) / run.info.run_id / "artifacts"


def test_list_run_paths(experiment: Experiment):
    from hydraflow.core.mlflow import list_run_paths

    dirs = list_run_paths(experiment.name, "artifacts")
    assert all(d.is_dir() for d in dirs)
