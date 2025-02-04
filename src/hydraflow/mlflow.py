"""Provide functionality to log parameters from Hydra configuration objects.

This module provides functions to log parameters from Hydra configuration objects
to MLflow, set experiments, and manage tracking URIs. It integrates Hydra's
configuration management with MLflow's experiment tracking capabilities.

Key Features:
- **Experiment Management**: Set experiment names and tracking URIs using Hydra
  configuration details.
- **Parameter Logging**: Log parameters from Hydra configuration objects to MLflow,
  supporting both synchronous and asynchronous logging.
- **Run Collection**: Utilize the `RunCollection` class to manage and interact with
  multiple MLflow runs, providing methods to filter and retrieve runs based on
  various criteria.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import mlflow
import mlflow.artifacts
from hydra.core.hydra_config import HydraConfig
from mlflow.entities import ViewType
from mlflow.tracking.fluent import SEARCH_MAX_RESULTS_PANDAS, _get_experiment_id

from hydraflow.config import iter_params
from hydraflow.run_collection import RunCollection

if TYPE_CHECKING:
    from mlflow.entities.experiment import Experiment


def set_experiment(
    prefix: str = "",
    suffix: str = "",
    uri: str | Path | None = None,
    name: str | None = None,
) -> Experiment:
    """Set the experiment name and tracking URI optionally.

    This function sets the experiment name by combining the given prefix,
    the job name from HydraConfig, and the given suffix. Optionally, it can
    also set the tracking URI.

    Args:
        prefix (str): The prefix to prepend to the experiment name.
        suffix (str): The suffix to append to the experiment name.
        uri (str | Path | None): The tracking URI to use. Defaults to None.
        name (str | None): The name of the experiment. Defaults to None.

    Returns:
        Experiment: An instance of `mlflow.entities.Experiment` representing
        the new active experiment.

    """
    if uri is not None:
        mlflow.set_tracking_uri(uri)

    if name is not None:
        return mlflow.set_experiment(name)

    hc = HydraConfig.get()
    name = f"{prefix}{hc.job.name}{suffix}"
    return mlflow.set_experiment(name)


def log_params(config: object, *, synchronous: bool | None = None) -> None:
    """Log the parameters from the given configuration object.

    This method logs the parameters from the provided configuration object
    using MLflow. It iterates over the parameters and logs them using the
    `mlflow.log_param` method.

    Args:
        config (object): The configuration object to log the parameters from.
        synchronous (bool | None): Whether to log the parameters synchronously.
            Defaults to None.

    """
    for key, value in iter_params(config):
        mlflow.log_param(key, value, synchronous=synchronous)


def search_runs(  # noqa: PLR0913
    *,
    experiment_ids: list[str] | None = None,
    filter_string: str = "",
    run_view_type: int = ViewType.ACTIVE_ONLY,
    max_results: int = SEARCH_MAX_RESULTS_PANDAS,
    order_by: list[str] | None = None,
    search_all_experiments: bool = False,
    experiment_names: list[str] | None = None,
) -> RunCollection:
    """Search for Runs that fit the specified criteria.

    This function wraps the `mlflow.search_runs` function and returns the
    results as a `RunCollection` object. It allows for flexible searching of
    MLflow runs based on various criteria.

    Note:
        The returned runs are sorted by their start time in ascending order.

    Args:
        experiment_ids (list[str] | None): List of experiment IDs. Search can
            work with experiment IDs or experiment names, but not both in the
            same call. Values other than ``None`` or ``[]`` will result in
            error if ``experiment_names`` is also not ``None`` or ``[]``.
            ``None`` will default to the active experiment if ``experiment_names``
            is ``None`` or ``[]``.
        filter_string (str): Filter query string, defaults to searching all
            runs.
        run_view_type (int): one of enum values ``ACTIVE_ONLY``, ``DELETED_ONLY``,
            or ``ALL`` runs defined in :py:class:`mlflow.entities.ViewType`.
        max_results (int): The maximum number of runs to put in the dataframe.
            Default is 100,000 to avoid causing out-of-memory issues on the user's
            machine.
        order_by (list[str] | None): List of columns to order by (e.g.,
            "metrics.rmse"). The ``order_by`` column can contain an optional
            ``DESC`` or ``ASC`` value. The default is ``ASC``. The default
            ordering is to sort by ``start_time DESC``, then ``run_id``.
            ``start_time DESC``, then ``run_id``.
        search_all_experiments (bool): Boolean specifying whether all
            experiments should be searched. Only honored if ``experiment_ids``
            is ``[]`` or ``None``.
        experiment_names (list[str] | None): List of experiment names. Search
            can work with experiment IDs or experiment names, but not both in
            the same call. Values other than ``None`` or ``[]`` will result in
            error if ``experiment_ids`` is also not ``None`` or ``[]``.
            ``experiment_ids`` is also not ``None`` or ``[]``. ``None`` will
            default to the active experiment if ``experiment_ids`` is ``None``
            or ``[]``.

    Returns:
        A `RunCollection` object containing the search results.

    """
    runs = mlflow.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        run_view_type=run_view_type,
        max_results=max_results,
        order_by=order_by,
        output_format="list",
        search_all_experiments=search_all_experiments,
        experiment_names=experiment_names,
    )
    runs = sorted(runs, key=lambda run: run.info.start_time)  # type: ignore
    return RunCollection(runs)  # type: ignore


def list_runs(
    experiment_names: str | list[str] | None = None,
    n_jobs: int = 0,
    status: str | list[str] | int | list[int] | None = None,
) -> RunCollection:
    """List all runs for the specified experiments.

    This function retrieves all runs for the given list of experiment names.
    If no experiment names are provided (None), it defaults to searching all runs
    for the currently active experiment. If an empty list is provided, the function
    will search all runs for all experiments except the "Default" experiment.
    The function returns the results as a `RunCollection` object.

    Note:
        The returned runs are sorted by their start time in ascending order.

    Args:
        experiment_names (list[str] | None): List of experiment names to search
            for runs. If None or an empty list is provided, the function will
            search the currently active experiment or all experiments except
            the "Default" experiment.
        n_jobs (int): The number of jobs to run in parallel. If 0, the function
            will search runs sequentially.
        status (str | list[str] | int | list[int] | None): The status of the runs
            to filter.

    Returns:
        RunCollection: A `RunCollection` instance containing the runs for the
        specified experiments.

    """
    rc = _list_runs(experiment_names, n_jobs)
    if status is None:
        return rc

    return rc.filter(status=status)


def _list_runs(
    experiment_names: str | list[str] | None = None,
    n_jobs: int = 0,
) -> RunCollection:
    if isinstance(experiment_names, str):
        experiment_names = [experiment_names]

    elif experiment_names == []:
        experiments = mlflow.search_experiments()
        experiment_names = [e.name for e in experiments if e.name != "Default"]

    if n_jobs == 0:
        return search_runs(experiment_names=experiment_names)

    if experiment_names is None:
        experiment_id = _get_experiment_id()
        experiment_names = [mlflow.get_experiment(experiment_id).name]

    run_ids = []

    for name in experiment_names:
        if experiment := mlflow.get_experiment_by_name(name):
            loc = experiment.artifact_location

            if isinstance(loc, str):
                if loc.startswith("file:"):
                    path = Path(mlflow.artifacts.download_artifacts(loc))
                elif Path(loc).is_dir():
                    path = Path(loc)
                else:
                    continue  # no cov

                run_ids.extend(file.stem for file in path.iterdir() if file.is_dir())

    it = (joblib.delayed(mlflow.get_run)(run_id) for run_id in run_ids)
    runs = joblib.Parallel(n_jobs, prefer="threads")(it)
    runs = sorted(runs, key=lambda run: run.info.start_time)  # type: ignore
    return RunCollection(runs)  # type: ignore
