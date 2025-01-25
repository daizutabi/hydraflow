import mlflow
import pytest
from mlflow.entities import Experiment

from hydraflow.run_collection import RunCollection


@pytest.fixture(scope="module")
def experiment(experiment_name: str):
    experiment = mlflow.set_experiment(experiment_name)

    for x in range(3):
        with mlflow.start_run(run_name=f"{x}"):
            pass

    return experiment


@pytest.fixture
def rc(experiment: Experiment):
    from hydraflow.mlflow import search_runs

    return search_runs(experiment_names=[experiment.name])


def test_info_run_id(rc: RunCollection):
    assert len(rc.info.run_id) == 3


def test_info_artifact_uri(rc: RunCollection):
    uri = rc.info.artifact_uri
    assert all(u.startswith("file://") for u in uri)  # type: ignore
    assert all(u.endswith("/artifacts") for u in uri)  # type: ignore


def test_info_artifact_dir(rc: RunCollection):
    dir = rc.info.artifact_dir
    assert all(d.stem == "artifacts" for d in dir)


def test_info_empty_run_collection():
    rc = RunCollection([])
    assert rc.info.run_id == []
    assert rc.info.artifact_uri == []
    assert rc.info.artifact_dir == []
