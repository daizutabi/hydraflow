import os
import uuid
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def chdir(tmp_path_factory: pytest.TempPathFactory):
    cwd = Path.cwd()
    name = str(uuid.uuid4())

    os.chdir(tmp_path_factory.mktemp(name, numbered=False))

    yield name

    os.chdir(cwd)


@pytest.fixture(scope="module")
def config(chdir):
    from hydraflow.jobs.io import load_config

    def config(text: str):
        Path("hydraflow.yaml").write_text(text)
        return load_config()

    return config
