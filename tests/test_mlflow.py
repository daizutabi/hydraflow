from pathlib import Path

import mlflow
import pytest
from mlflow.entities import Experiment, Run, RunStatus


@pytest.fixture(scope="module")
def experiment(experiment_name: str):
    from hydraflow.mlflow import log_params, set_experiment

    experiment = set_experiment(uri="mlruns", name="e")

    with mlflow.start_run():
        log_params({"name": experiment_name})

    mlflow.start_run()
    mlflow.end_run(status=RunStatus.to_string(RunStatus.RUNNING))

    mlflow.start_run()
    mlflow.end_run(status=RunStatus.to_string(RunStatus.FAILED))

    yield experiment

    mlflow.set_tracking_uri("")


def test_set_experiment_uri(experiment: Experiment):
    assert mlflow.get_tracking_uri() == "mlruns"


def test_set_experiment_location(experiment: Experiment):
    path = Path.cwd() / "mlruns" / experiment.experiment_id
    loc = experiment.artifact_location
    assert isinstance(loc, str)
    loc = loc.replace("file:", "")  # for windows
    assert Path(loc) == path


def test_set_experiment_name(experiment: Experiment):
    e = mlflow.get_experiment_by_name("e")
    assert e
    assert e.experiment_id == experiment.experiment_id


def test_search_runs(experiment: Experiment):
    from hydraflow.mlflow import search_runs

    rc = search_runs(experiment_names=[experiment.name])
    assert len(rc) == 3


@pytest.fixture(scope="module")
def run(experiment: Experiment):
    from hydraflow.mlflow import search_runs

    rc = search_runs(experiment_names=[experiment.name])
    return rc.first()


def test_log_params(run: Run, experiment_name):
    assert run.data.params["name"] == experiment_name


def test_get_artifact_dir_from_utils(run: Run, experiment: Experiment):
    from hydraflow.utils import get_artifact_dir

    loc = experiment.artifact_location
    assert isinstance(loc, str)
    loc = loc.replace("file:", "")  # for windows
    path = Path(loc) / run.info.run_id / "artifacts"
    assert get_artifact_dir(run) == path


@pytest.mark.parametrize(
    ("status", "n"),
    [
        (RunStatus.FINISHED, 1),
        (RunStatus.RUNNING, 1),
        (RunStatus.FAILED, 1),
        (None, 3),
    ],
)
@pytest.mark.parametrize("n_jobs", [0, 1, 2])
@pytest.mark.parametrize("func", [lambda x: x, list, lambda x: [], lambda x: None])
def test_list_runs(experiment: Experiment, status, n, n_jobs, func):
    from hydraflow.mlflow import list_runs

    experiment_names = func(experiment.name)
    rc = list_runs(experiment_names=experiment_names, status=status, n_jobs=n_jobs)
    assert len(rc) == n
