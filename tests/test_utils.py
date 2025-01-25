import mlflow
import pytest

from hydraflow.run_collection import RunCollection


@pytest.fixture(scope="module")
def experiment_name(experiment_name: str):
    for x in range(4):
        with mlflow.start_run(run_name=f"{x}"):
            pass

    yield experiment_name


@pytest.fixture
def rc(experiment_name: str):
    from hydraflow.mlflow import search_runs

    return search_runs(experiment_names=[experiment_name])


def test_remove_run(rc: RunCollection):
    from hydraflow.utils import get_artifact_dir, remove_run

    paths = [get_artifact_dir(r).parent for r in rc]

    assert all(path.exists() for path in paths)

    remove_run(rc)

    assert not any(path.exists() for path in paths)
