"""Integration of MLflow experiment tracking with Hydra configuration management.

This module provides functions to log parameters from Hydra configuration objects
to MLflow, set experiments, and manage tracking URIs. It integrates Hydra's
configuration management with MLflow's experiment tracking capabilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import joblib
import mlflow
import mlflow.artifacts

from hydraflow.config import iter_params
from hydraflow.run_collection import RunCollection
from hydraflow.utils import file_uri_to_path, get_artifact_dir

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


def log_params(config: Any, *, synchronous: bool | None = None) -> None:
    """Log the parameters from the given configuration object.

    This method logs the parameters from the provided configuration object
    using MLflow. It iterates over the parameters and logs them using the
    `mlflow.log_param` method.

    Args:
        config (Any): The configuration object to log the parameters from.
        synchronous (bool | None): Whether to log the parameters synchronously.
            Defaults to None.

    """
    for key, value in iter_params(config):
        mlflow.log_param(key, value, synchronous=synchronous)


def log_text(from_dir: Path, pattern: str = "*.log") -> None:
    """Log text files in the given directory as artifacts.

    Append the text files to the existing text file in the artifact directory.

    Args:
        from_dir (Path): The directory to find the logs in.
        pattern (str): The pattern to match the logs.

    """
    artifact_dir = get_artifact_dir()

    for file in from_dir.glob(pattern):
        if not file.is_file():
            continue

        file_artifact = artifact_dir / file.name
        if file_artifact.exists():
            text = file_artifact.read_text()
            if not text.endswith("\n"):
                text += "\n"
        else:
            text = ""

        text += file.read_text()
        mlflow.log_text(text, file.name)


def list_run_paths(
    experiment_names: str | list[str] | None = None,
    *other: str,
) -> list[Path]:
    """List all run paths for the specified experiments.

    This function retrieves all run paths for the given list of experiment names.
    If no experiment names are provided (None), the function will search all runs
    for all experiments except the "Default" experiment.

    Args:
        experiment_names (list[str] | None): List of experiment names to search
            for runs. If None is provided, the function will search all runs
            for all experiments except the "Default" experiment.
        *other (str): The parts of the run directory to join.

    Returns:
        list[Path]: A list of run paths for the specified experiments.

    """
    if isinstance(experiment_names, str):
        experiment_names = [experiment_names]

    elif experiment_names is None:
        experiments = mlflow.search_experiments()
        experiment_names = [e.name for e in experiments if e.name != "Default"]

    run_paths: list[Path] = []

    for name in experiment_names:
        if experiment := mlflow.get_experiment_by_name(name):
            uri = experiment.artifact_location

            if isinstance(uri, str):
                path = file_uri_to_path(uri)
                run_paths.extend(p for p in path.iterdir() if p.is_dir())

    if other:
        return [p.joinpath(*other) for p in run_paths]

    return run_paths


def list_run_ids(experiment_names: str | list[str] | None = None) -> list[str]:
    """List all run IDs for the specified experiments.

    This function retrieves all runs for the given list of experiment names.
    If no experiment names are provided (None), the function will search all
    runs for all experiments except the "Default" experiment.

    Args:
        experiment_names (list[str] | None): List of experiment names to search
            for runs. If None is provided, the function will search all runs
            for all experiments except the "Default" experiment.

    Returns:
        list[str]: A list of run IDs for the specified experiments.

    """
    return [run_path.stem for run_path in list_run_paths(experiment_names)]


def list_runs(
    experiment_names: str | list[str] | None = None,
    n_jobs: int = 0,
) -> RunCollection:
    """List all runs for the specified experiments.

    This function retrieves all runs for the given list of experiment names.
    If no experiment names are provided (None), the function will search all runs
    for all experiments except the "Default" experiment.
    The function returns the results as a `RunCollection` object.

    Note:
        The returned runs are sorted by their start time in ascending order.

    Args:
        experiment_names (list[str] | None): List of experiment names to search
            for runs. If None is provided, the function will search all runs
            for all experiments except the "Default" experiment.
        n_jobs (int): The number of jobs to retrieve runs in parallel.

    Returns:
        RunCollection: A `RunCollection` instance containing the runs for the
        specified experiments.

    """
    run_ids = list_run_ids(experiment_names)

    if n_jobs == 0:
        runs = [mlflow.get_run(run_id) for run_id in run_ids]

    else:
        it = (joblib.delayed(mlflow.get_run)(run_id) for run_id in run_ids)
        runs = joblib.Parallel(n_jobs, backend="threading")(it)

    runs = sorted(runs, key=lambda run: run.info.start_time)  # type: ignore
    return RunCollection(runs)  # type: ignore
