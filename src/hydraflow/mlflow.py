"""
This module provides functionality to log parameters from Hydra
configuration objects and set up experiments using MLflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
from hydra.core.hydra_config import HydraConfig
from mlflow.entities import ViewType
from mlflow.tracking.fluent import SEARCH_MAX_RESULTS_PANDAS

from hydraflow.config import iter_params
from hydraflow.run_collection import RunCollection

if TYPE_CHECKING:
    from mlflow.entities.experiment import Experiment


def set_experiment(
    prefix: str = "",
    suffix: str = "",
    uri: str | Path | None = None,
) -> Experiment:
    """
    Set the experiment name and tracking URI optionally.

    This function sets the experiment name by combining the given prefix,
    the job name from HydraConfig, and the given suffix. Optionally, it can
    also set the tracking URI.

    Args:
        prefix (str): The prefix to prepend to the experiment name.
        suffix (str): The suffix to append to the experiment name.
        uri (str | Path | None): The tracking URI to use. Defaults to None.

    Returns:
        Experiment: An instance of `mlflow.entities.Experiment` representing
        the new active experiment.
    """
    if uri is not None:
        mlflow.set_tracking_uri(uri)

    hc = HydraConfig.get()
    name = f"{prefix}{hc.job.name}{suffix}"
    return mlflow.set_experiment(name)


def log_params(config: object, *, synchronous: bool | None = None) -> None:
    """
    Log the parameters from the given configuration object.

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


def search_runs(
    experiment_ids: list[str] | None = None,
    filter_string: str = "",
    run_view_type: int = ViewType.ACTIVE_ONLY,
    max_results: int = SEARCH_MAX_RESULTS_PANDAS,
    order_by: list[str] | None = None,
    search_all_experiments: bool = False,
    experiment_names: list[str] | None = None,
) -> RunCollection:
    """
    Search for Runs that fit the specified criteria.

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


def list_runs(experiment_names: list[str] | None = None) -> RunCollection:
    """
    List all runs for the specified experiments.

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

    Returns:
        A `RunCollection` object containing the runs for the specified experiments.
    """
    if experiment_names == []:
        experiments = mlflow.search_experiments()
        experiment_names = [e.name for e in experiments if e.name != "Default"]

    return search_runs(experiment_names=experiment_names)
