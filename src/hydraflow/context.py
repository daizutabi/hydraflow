"""
This module provides context managers to log parameters and manage the MLflow
run context.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
from hydra.core.hydra_config import HydraConfig
from watchdog.events import FileModifiedEvent, PatternMatchingEventHandler
from watchdog.observers import Observer

from hydraflow.info import get_artifact_dir
from hydraflow.mlflow import log_params

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from mlflow.entities.run import Run

log = logging.getLogger(__name__)


@contextmanager
def log_run(
    config: object,
    *,
    synchronous: bool | None = None,
) -> Iterator[None]:
    """
    Log the parameters from the given configuration object and manage the MLflow
    run context.

    This context manager logs the parameters from the provided configuration object
    using MLflow. It also manages the MLflow run context, ensuring that artifacts
    are logged and the run is properly closed.

    Args:
        config (object): The configuration object to log the parameters from.
        synchronous (bool | None): Whether to log the parameters synchronously.
            Defaults to None.

    Yields:
        None

    Example:
        ```python
        with log_run(config):
            # Perform operations within the MLflow run context
            pass
        ```
    """
    log_params(config, synchronous=synchronous)

    hc = HydraConfig.get()
    output_dir = Path(hc.runtime.output_dir)

    # Save '.hydra' config directory first.
    output_subdir = output_dir / (hc.output_subdir or "")
    mlflow.log_artifacts(output_subdir.as_posix(), hc.output_subdir)

    def log_artifact(path: Path) -> None:
        local_path = (output_dir / path).as_posix()
        mlflow.log_artifact(local_path)

    try:
        with watch(log_artifact, output_dir, ignore_log=False):
            yield

    except Exception as e:
        log.error(f"Error during log_run: {e}")
        raise

    finally:
        # Save output_dir including '.hydra' config directory.
        mlflow.log_artifacts(output_dir.as_posix())


@contextmanager
def start_run(
    config: object,
    *,
    run_id: str | None = None,
    experiment_id: str | None = None,
    run_name: str | None = None,
    nested: bool = False,
    parent_run_id: str | None = None,
    tags: dict[str, str] | None = None,
    description: str | None = None,
    log_system_metrics: bool | None = None,
    synchronous: bool | None = None,
) -> Iterator[Run]:
    """
    Start an MLflow run and log parameters using the provided configuration object.

    This context manager starts an MLflow run and logs parameters using the specified
    configuration object. It ensures that the run is properly closed after completion.

    Args:
        config (object): The configuration object to log parameters from.
        run_id (str | None): The existing run ID. Defaults to None.
        experiment_id (str | None): The experiment ID. Defaults to None.
        run_name (str | None): The name of the run. Defaults to None.
        nested (bool): Whether to allow nested runs. Defaults to False.
        parent_run_id (str | None): The parent run ID. Defaults to None.
        tags (dict[str, str] | None): Tags to associate with the run. Defaults to None.
        description (str | None): A description of the run. Defaults to None.
        log_system_metrics (bool | None): Whether to log system metrics. Defaults to None.
        synchronous (bool | None): Whether to log parameters synchronously. Defaults to None.

    Yields:
        Run: An MLflow Run object representing the started run.

    Example:
        with start_run(config) as run:
            # Perform operations within the MLflow run context
            pass

    See Also:
        - `mlflow.start_run`: The MLflow function to start a run directly.
        - `log_run`: A context manager to log parameters and manage the MLflow
           run context.
    """
    with mlflow.start_run(
        run_id=run_id,
        experiment_id=experiment_id,
        run_name=run_name,
        nested=nested,
        parent_run_id=parent_run_id,
        tags=tags,
        description=description,
        log_system_metrics=log_system_metrics,
    ) as run:
        with log_run(config, synchronous=synchronous):
            yield run


@contextmanager
def watch(
    callback: Callable[[Path], None],
    dir: Path | str = "",
    timeout: int = 60,
    ignore_patterns: list[str] | None = None,
    ignore_log: bool = True,
) -> Iterator[None]:
    """
    Watch the given directory for changes and call the provided function
    when a change is detected.

    This context manager sets up a file system watcher on the specified directory.
    When a file modification is detected, the provided function is called with
    the path of the modified file. The watcher runs for the specified timeout
    period or until the context is exited.

    Args:
        callback (Callable[[Path], None]): The function to call when a change is
            detected. It should accept a single argument of type `Path`,
            which is the path of the modified file.
        dir (Path | str): The directory to watch. If not specified,
            the current MLflow artifact URI is used. Defaults to "".
        timeout (int): The timeout period in seconds for the watcher
            to run after the context is exited. Defaults to 60.

    Yields:
        None

    Example:
        ```python
        with watch(log_artifact, "/path/to/dir"):
            # Perform operations while watching the directory for changes
            pass
        ```
    """
    dir = dir or get_artifact_dir()
    if isinstance(dir, Path):
        dir = dir.as_posix()

    handler = Handler(callback, ignore_patterns=ignore_patterns, ignore_log=ignore_log)
    observer = Observer()
    observer.schedule(handler, dir, recursive=True)
    observer.start()

    try:
        yield

    except Exception as e:
        log.error(f"Error during watch: {e}")
        raise

    finally:
        elapsed = 0
        while not observer.event_queue.empty():
            time.sleep(0.2)
            elapsed += 0.2
            if elapsed > timeout:
                break

        observer.stop()
        observer.join()


class Handler(PatternMatchingEventHandler):
    def __init__(
        self,
        func: Callable[[Path], None],
        ignore_patterns: list[str] | None = None,
        ignore_log: bool = True,
    ) -> None:
        self.func = func

        if ignore_log:
            if ignore_patterns:
                ignore_patterns.append("*.log")
            else:
                ignore_patterns = ["*.log"]

        super().__init__(ignore_patterns=ignore_patterns)

    def on_modified(self, event: FileModifiedEvent) -> None:
        file = Path(str(event.src_path))
        if file.is_file():
            self.func(file)


@contextmanager
def chdir_artifact(
    run: Run,
    artifact_path: str | None = None,
) -> Iterator[Path]:
    """
    Change the current working directory to the artifact directory of the
    given run.

    This context manager changes the current working directory to the artifact
    directory of the given run. It ensures that the directory is changed back
    to the original directory after the context is exited.

    Args:
        run (Run): The run to get the artifact directory from.
        artifact_path (str | None): The artifact path.
    """
    curdir = Path.cwd()
    path = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id,
        artifact_path=artifact_path,
    )

    os.chdir(path)
    try:
        yield Path(path)

    finally:
        os.chdir(curdir)
