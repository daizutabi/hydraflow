import os
import uuid
from pathlib import Path

import mlflow
import pytest


@pytest.fixture(scope="module")
def experiment_name(tmp_path_factory: pytest.TempPathFactory):
    cwd = Path.cwd()
    name = str(uuid.uuid4())
    os.chdir(tmp_path_factory.mktemp(name))
    mlflow.set_experiment(name)
    yield name
    os.chdir(cwd)
