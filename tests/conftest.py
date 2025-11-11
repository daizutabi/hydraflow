import os
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from hydraflow.core.io import iter_artifacts_dirs


@pytest.fixture(scope="module")
def chdir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    cwd = Path()
    name = str(uuid.uuid4())

    os.chdir(tmp_path_factory.mktemp(name, numbered=False))

    yield Path().absolute()

    os.chdir(cwd)


@pytest.fixture(scope="module")
def experiment_name(chdir: Path) -> str:
    return chdir.name


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
def list_artifacts_dirs(run_script):
    def artifacts_dirs(filename: Path | str, args: list[str]):
        experiment_name = run_script(filename, args)
        return list(iter_artifacts_dirs("mlruns", experiment_name))

    return artifacts_dirs


def load(path: Path) -> DictConfig:
    config_file = path / ".hydra/config.yaml"
    return OmegaConf.load(config_file)  # pyright: ignore[reportReturnType]


@pytest.fixture(scope="module")
def collect(list_artifacts_dirs):
    def collect(filename: Path | str, args: list[str]):
        artifacts_dirs = list_artifacts_dirs(filename, args)
        configs = [load(artifacts_dir) for artifacts_dir in artifacts_dirs]
        return list(zip(artifacts_dirs, configs, strict=True))

    return collect
