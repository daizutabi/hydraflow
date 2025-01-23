from __future__ import annotations

import mlflow
import pytest

from hydraflow.run_collection import RunCollection


@pytest.fixture
def rc(monkeypatch, tmp_path):
    from hydraflow.mlflow import search_runs

    monkeypatch.chdir(tmp_path)

    mlflow.set_experiment("test_run")
    for x in range(4):
        with mlflow.start_run(run_name=f"{x}"):
            pass

    x = search_runs()
    assert isinstance(x, RunCollection)
    yield x


def test_remove_run(rc: RunCollection):
    from hydraflow.utils import get_artifact_dir, remove_run

    paths = [get_artifact_dir(r).parent for r in rc]

    assert all(path.exists() for path in paths)

    remove_run(rc)

    assert not any(path.exists() for path in paths)
