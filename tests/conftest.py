import importlib
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore


@pytest.fixture(scope="module")
def experiment_name(tmp_path_factory: pytest.TempPathFactory):
    cwd = Path.cwd()
    name = str(uuid.uuid4())

    os.chdir(tmp_path_factory.mktemp(name))

    yield name

    os.chdir(cwd)


@pytest.fixture(scope="module")
def run_script(experiment_name: str):
    parent = Path(__file__).parent

    def run_script(filename: str, args: list[str]):
        file = parent / filename
        job_name = f"hydra.job.name={experiment_name}"

        args = [sys.executable, file.as_posix(), *args, job_name]
        subprocess.check_call(args)

        return experiment_name

    return run_script


@pytest.fixture(scope="module")
def collect(run_script):
    from hydraflow.mlflow import search_runs

    def collect(filename: str, args: list[str]):
        experiment_name = run_script(filename, args)
        return search_runs(experiment_names=[experiment_name])

    return collect


@pytest.fixture(scope="module")
def get_config_class():
    parent = Path(__file__).parent

    def get_config_class(filename: str):
        file = parent / filename

        sys.path.insert(0, file.parent.as_posix())
        module = importlib.import_module(file.stem)
        del sys.path[0]

        return module.Config

    return get_config_class


@pytest.fixture
def get_config(get_config_class):
    cs = ConfigStore.instance()

    def get_config(filename: str, overrides: list[str] | None = None):
        cls = get_config_class(filename)

        name = str(uuid.uuid4())
        cs.store(name=name, node=cls)

        with initialize(version_base=None):
            return compose(config_name=name, overrides=overrides)

    return get_config
