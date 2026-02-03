"""Provide context managers to log parameters and manage the MLflow run context."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from hydra.core.hydra_config import HydraConfig

from .io import get_artifact_dir, log_text

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mlflow.entities.run import Run

logger = logging.getLogger(__name__)


@contextmanager
def log_run(run: Run) -> Iterator[None]:
    """Log the parameters from the given configuration instance.

    This context manager logs the parameters from the provided configuration instance
    using MLflow. It also manages the MLflow run context, ensuring that artifacts
    are logged and the run is properly closed.

    Args:
        run (Run): The run instance.

    Yields:
        None

    """
    import mlflow

    hc = HydraConfig.get()
    hydra_output_dir = Path(hc.runtime.output_dir)

    # Save '.hydra' config directory.
    hydra_dir = hydra_output_dir / (hc.output_subdir or "")
    mlflow.log_artifacts(hydra_dir.as_posix(), ".hydra")

    try:
        yield

    except Exception as e:
        msg = f"Error during log_run: {e}"
        logger.exception(msg)
        raise

    finally:
        log_text(run, hydra_output_dir)


@contextmanager
def start_run(
    *,
    chdir: bool = False,
    run_id: str | None = None,
    experiment_id: str | None = None,
    run_name: str | None = None,
    nested: bool = False,
    parent_run_id: str | None = None,
    tags: dict[str, str] | None = None,
    description: str | None = None,
    log_system_metrics: bool | None = None,
) -> Iterator[Run]:
    """Start an MLflow run and log parameters using the provided configuration instance.

    This context manager starts an MLflow run and logs parameters using the specified
    configuration instance. It ensures that the run is properly closed after completion.

    Args:
        chdir (bool): Whether to change the current working directory to the
            artifact directory of the current run. Defaults to False.
        run_id (str | None): The existing run ID. Defaults to None.
        experiment_id (str | None): The experiment ID. Defaults to None.
        run_name (str | None): The name of the run. Defaults to None.
        nested (bool): Whether to allow nested runs. Defaults to False.
        parent_run_id (str | None): The parent run ID. Defaults to None.
        tags (dict[str, str] | None): Tags to associate with the run. Defaults to None.
        description (str | None): A description of the run. Defaults to None.
        log_system_metrics (bool | None): Whether to log system metrics.
            Defaults to None.

    Yields:
        Run: An MLflow Run instance representing the started run.

    """
    import mlflow

    with (
        mlflow.start_run(
            run_id=run_id,
            experiment_id=experiment_id,
            run_name=run_name,
            nested=nested,
            parent_run_id=parent_run_id,
            tags=tags,
            description=description,
            log_system_metrics=log_system_metrics,
        ) as run,
        log_run(run),
    ):
        if chdir:
            with chdir_artifact(run):
                yield run
        else:
            yield run


@contextmanager
def chdir_artifact(run: Run) -> Iterator[Path]:
    """Change the current working directory to the artifact directory of the given run.

    This context manager changes the current working directory to the artifact
    directory of the given run. It ensures that the directory is changed back
    to the original directory after the context is exited.

    Args:
        run (Run | None): The run to get the artifact directory from.

    Yields:
        str: The artifact directory

    """
    current_dir = Path.cwd()
    artifact_dir = get_artifact_dir(run)

    try:
        os.chdir(artifact_dir)
        yield artifact_dir

    finally:
        os.chdir(current_dir)
