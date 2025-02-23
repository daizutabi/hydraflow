import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def chdir(tmp_path_factory: pytest.TempPathFactory):
    cwd = Path.cwd()
    name = str(uuid.uuid4())

    os.chdir(tmp_path_factory.mktemp(name, numbered=False))

    yield

    os.chdir(cwd)


@pytest.fixture(scope="module")
def experiment_name(chdir):
    return Path.cwd().name


@pytest.fixture(scope="module")
def run_script(experiment_name: str):
    parent = Path(__file__).parent

    def run_script(filename: Path | str, args: list[str]):
        file = parent / filename
        job_name = f"hydra.job.name={experiment_name}"

        args = [sys.executable, file.as_posix(), *args, job_name]
        subprocess.run(args, check=False)

        return experiment_name

    return run_script


@pytest.fixture(scope="module")
def collect(run_script):
    from hydraflow.core.mlflow import list_runs

    def collect(filename: Path | str, args: list[str]):
        experiment_name = run_script(filename, args)
        return list_runs(experiment_name)

    return collect
