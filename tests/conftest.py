from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import pytest
from omegaconf import DictConfig, OmegaConf

from hydraflow.core.io import iter_artifacts_dirs

if TYPE_CHECKING:
    from collections.abc import Callable


def load(path: Path) -> DictConfig:
    config_file = path / ".hydra/config.yaml"
    return OmegaConf.load(config_file)  # pyright: ignore[reportReturnType]


type Results = list[tuple[Path, DictConfig]]
type Collect = Callable[..., Results]


@pytest.fixture(scope="module")
def collect(tmp_path_factory: pytest.TempPathFactory) -> Collect:
    def collect(filename: Path, *args: list[str]) -> Results:
        orig_dir = Path().absolute()

        job_name = str(uuid.uuid4())
        os.chdir(tmp_path_factory.mktemp(job_name))

        job_name_arg = f"hydra.job.name={job_name}"

        try:
            for arg in args:
                args_ = [sys.executable, filename.as_posix(), *arg, job_name_arg]
                subprocess.run(args_, check=False)

            if Path("mlflow.db").exists():
                mlflow.set_tracking_uri("sqlite:///mlflow.db")
            else:
                mlflow.set_tracking_uri("mlruns")

            artifacts_dirs = list(iter_artifacts_dirs(job_name))
            configs = [load(artifacts_dir) for artifacts_dir in artifacts_dirs]

            return list(zip(artifacts_dirs, configs, strict=True))

        finally:
            os.chdir(orig_dir)

    return collect
