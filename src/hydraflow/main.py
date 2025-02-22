"""Integration of MLflow experiment tracking with Hydra configuration management.

This module provides decorators and utilities to seamlessly combine Hydra's
configuration management with MLflow's experiment tracking capabilities. It
enables automatic run deduplication, configuration storage, and experiment
management.

The main functionality is provided through the `main` decorator, which can be
used to wrap experiment entry points. This decorator handles:
- Configuration management via Hydra
- Experiment tracking via MLflow
- Run deduplication based on configurations
- Working directory management
- Automatic configuration storage

Example:
    ```python
    from dataclasses import dataclass
    from mlflow.entities import Run

    @dataclass
    class Config:
        learning_rate: float
        batch_size: int

    @main(Config)
    def train(run: Run, config: Config):
        # Your training code here
        pass
    ```

"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, TypeVar

import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from mlflow.entities import RunStatus
from omegaconf import OmegaConf

import hydraflow
from hydraflow.utils import file_uri_to_path

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mlflow.entities import Run

FINISHED = RunStatus.to_string(RunStatus.FINISHED)

T = TypeVar("T")


def main(
    node: T | type[T],
    config_name: str = "config",
    *,
    chdir: bool = False,
    force_new_run: bool = False,
    match_overrides: bool = False,
    rerun_finished: bool = False,
):
    """Decorator for configuring and running MLflow experiments with Hydra.

    This decorator combines Hydra configuration management with MLflow experiment
    tracking. It automatically handles run deduplication and configuration storage.

    Args:
        node: Configuration node class or instance defining the structure of the
            configuration.
        config_name: Name of the configuration. Defaults to "config".
        chdir: If True, changes working directory to the artifact directory
            of the run. Defaults to False.
        force_new_run: If True, always creates a new MLflow run instead of
            reusing existing ones. Defaults to False.
        match_overrides: If True, matches runs based on Hydra CLI overrides
            instead of full config. Defaults to False.
        rerun_finished: If True, allows rerunning completed runs. Defaults to
            False.

    """

    def decorator(app: Callable[[Run, T], None]) -> Callable[[], None]:
        ConfigStore.instance().store(config_name, node)

        @hydra.main(config_name=config_name, version_base=None)
        @wraps(app)
        def inner_decorator(config: T) -> None:
            hc = HydraConfig.get()
            experiment = mlflow.set_experiment(hc.job.name)

            if force_new_run:
                run_id = None
            else:
                uri = experiment.artifact_location
                overrides = hc.overrides.task if match_overrides else None
                run_id = get_run_id(uri, config, overrides)

                if run_id and not rerun_finished:
                    run = mlflow.get_run(run_id)
                    if run.info.status == FINISHED:
                        return

            with hydraflow.start_run(config, run_id=run_id, chdir=chdir) as run:
                app(run, config)

        return inner_decorator

    return decorator


def get_run_id(uri: str, config: object, overrides: list[str] | None) -> str | None:
    """Try to get the run ID for the given configuration.

    If the run is not found, the function will return None.

    Args:
        uri (str): The URI of the experiment.
        config (object): The configuration object.
        overrides (list[str] | None): The task overrides.

    Returns:
        The run ID for the given configuration or overrides. Returns None if
        no run ID is found.

    """
    for run_dir in file_uri_to_path(uri).iterdir():
        if run_dir.is_dir() and equals(run_dir, config, overrides):
            return run_dir.name

    return None


def equals(run_dir: Path, config: object, overrides: list[str] | None) -> bool:
    """Check if the run directory matches the given configuration or overrides.

    Args:
        run_dir (Path): The run directory.
        config (object): The configuration object.
        overrides (list[str] | None): The task overrides.

    Returns:
        True if the run directory matches the given configuration or overrides,
        False otherwise.

    """
    if overrides is None:
        path = run_dir / "artifacts/.hydra/config.yaml"
    else:
        path = run_dir / "artifacts/.hydra/overrides.yaml"
        config = overrides

    if not path.exists():
        return False

    return OmegaConf.load(path) == config
