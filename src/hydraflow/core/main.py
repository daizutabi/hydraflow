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
    import hydraflow
    from dataclasses import dataclass
    from mlflow.entities import Run

    @dataclass
    class Config:
        learning_rate: float
        batch_size: int

    @hydraflow.main(Config)
    def train(run: Run, config: Config):
        # Your training code here
        pass
    ```

"""

from __future__ import annotations

import logging
import sys
from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
from filelock import FileLock
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf

from hydraflow.core.context import start_run
from hydraflow.core.io import file_uri_to_path

if TYPE_CHECKING:
    from collections.abc import Callable

    from hydra.conf import HydraConf
    from mlflow.entities import Run
    from mlflow.entities.experiment import Experiment

logger = logging.getLogger("hydraflow")


def main[C](
    node: C | type[C],
    config_name: str = "config",
    *,
    tracking_uri: str | None = None,
    chdir: bool = False,
    force_new_run: bool = False,
    match_overrides: bool = False,
    rerun_finished: bool = False,
    dry_run: bool = False,
    update: Callable[[C], C | None] | None = None,
):
    """Decorator for configuring and running MLflow experiments with Hydra.

    This decorator combines Hydra configuration management with MLflow experiment
    tracking. It automatically handles run deduplication and configuration storage.

    Args:
        node: Configuration node class or instance defining the structure of the
            configuration.
        config_name: Name of the configuration. Defaults to "config".
        tracking_uri: The tracking URI for MLflow. If not provided, MLflow's
            default tracking URI is used. Defaults to None.
        chdir: If True, changes working directory to the artifact directory
            of the run. Defaults to False.
        force_new_run: If True, always creates a new MLflow run instead of
            reusing existing ones. Defaults to False.
        match_overrides: If True, matches runs based on Hydra CLI overrides
            instead of full config. Defaults to False.
        rerun_finished: If True, allows rerunning completed runs. Defaults to
            False.
        dry_run: If True, starts the hydra job but does not run the application
            itself. This allows users to preview the configuration and
            settings without executing the actual run. Defaults to False.
        update: A function that takes a configuration and returns a new
            configuration or None. The function can modify the configuration in-place
            and/or return it. If the function returns None, the original (potentially
            modified) configuration is used. Changes made by this function are saved
            to the configuration file. This is useful for adding derived parameters,
            ensuring consistency between related values, or adding runtime information
            to the configuration. Defaults to None.

    """
    import mlflow
    from mlflow.entities import RunStatus

    if "--dry-run" in sys.argv:
        dry_run = True
        sys.argv.remove("--dry-run")

    finished = RunStatus.to_string(RunStatus.FINISHED)  # pyright: ignore[reportUnknownMemberType]

    def decorator(app: Callable[[Run, C], None]) -> Callable[[], None]:
        ConfigStore.instance().store(config_name, node)

        @hydra.main(config_name=config_name, version_base=None)
        @wraps(app)
        def inner_decorator(cfg: C) -> None:
            hc = HydraConfig.get()

            if update:
                if cfg_ := update(cfg):
                    cfg = cfg_

                hydra_dir = Path(hc.runtime.output_dir) / (hc.output_subdir or "")
                cfg_path = hydra_dir.joinpath("config.yaml")
                OmegaConf.save(cfg, cfg_path)

            if dry_run:
                logger.info("Dry run:\n%s", OmegaConf.to_yaml(cfg).rstrip())
                return

            experiment = set_experiment(hc, tracking_uri)

            if force_new_run:
                run_id = None
            else:
                uri = experiment.artifact_location
                overrides = hc.overrides.task if match_overrides else None
                run_id = get_run_id(uri, cfg, overrides)  # pyright: ignore[reportUnknownArgumentType]

                if run_id and not rerun_finished:
                    run = mlflow.get_run(run_id)
                    if run.info.status == finished:  # pyright: ignore[reportUnknownMemberType]
                        return

            with start_run(run_id=run_id, chdir=chdir) as run:
                app(run, cfg)

        return inner_decorator

    return decorator


def set_experiment(hc: HydraConf, tracking_uri: str | None) -> Experiment:
    """Set the MLflow tracking URI if provided and experiment.

    Args:
        hc (HydraConf): The Hydra configuration instance.
        tracking_uri (str | None): The tracking URI for MLflow. If None,
            MLflow's default tracking URI is used.

    Returns:
        Experiment: The MLflow experiment instance.

    """
    import mlflow

    lock_manager = nullcontext()

    if hc.mode == RunMode.MULTIRUN:
        # Use the parent of sweep_dir for a global lock.
        # This handles race conditions across different sweeps
        # under a common parent.
        global_lock_dir = Path(hc.sweep.dir).parent
        # Ensure the directory exists before creating the lock file
        global_lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = global_lock_dir / ".mlflow.lock"
        lock_manager = FileLock(lock_path)

    with lock_manager:
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        return mlflow.set_experiment(hc.job.name)


def get_run_id(uri: str, config: Any, overrides: list[str] | None) -> str | None:
    """Try to get the run ID for the given configuration.

    If the run is not found, the function will return None.

    Args:
        uri (str): The URI of the experiment.
        config (object): The configuration instance.
        overrides (list[str] | None): The task overrides.

    Returns:
        The run ID for the given configuration or overrides. Returns None if
        no run ID is found.

    """
    path = file_uri_to_path(uri)
    if not path.is_dir():
        return None

    for run_dir in path.iterdir():
        if run_dir.is_dir() and equals(run_dir, config, overrides):
            return run_dir.name

    return None


def equals(run_dir: Path, config: Any, overrides: list[str] | None) -> bool:
    """Check if the run directory matches the given configuration or overrides.

    Args:
        run_dir (Path): The run directory.
        config (object): The configuration instance.
        overrides (list[str] | None): The task overrides.

    Returns:
        True if the run directory matches the given configuration or overrides,
        False otherwise.

    """
    if overrides is None:
        path = run_dir / "artifacts/.hydra/config.yaml"

        if not path.exists():
            return False

        return OmegaConf.load(path) == config

    path = run_dir / "artifacts/.hydra/overrides.yaml"

    if not path.exists():
        return False

    return sorted(OmegaConf.load(path)) == sorted(overrides)
