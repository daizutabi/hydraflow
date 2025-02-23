import os
import uuid
from pathlib import Path

import pytest

from hydraflow.jobs.conf import HydraflowConf


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
        cfg = load_config()
        Path("hydraflow.yaml").unlink()
        return cfg

    return config


@pytest.fixture(scope="module")
def args(config):
    def args(text: str):
        text = f"jobs:\n  a:\n    steps:\n      - args: {text}\n"
        cfg: HydraflowConf = config(text)
        return cfg.jobs["a"].steps[0].args

    return args
