"""
This module provides functionality for managing and interacting with MLflow runs.
It includes the `Runs` class and various methods to filter runs, retrieve run information,
log artifacts, and load configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from itertools import chain
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.entities import ViewType
from mlflow.entities.run import Run
from mlflow.tracking.fluent import SEARCH_MAX_RESULTS_PANDAS
from omegaconf import DictConfig, OmegaConf

from hydraflow.config import iter_params

if TYPE_CHECKING:
    from typing import Any


def search_runs(
    experiment_ids: list[str] | None = None,
    filter_string: str = "",
    run_view_type: int = ViewType.ACTIVE_ONLY,
    max_results: int = SEARCH_MAX_RESULTS_PANDAS,
    order_by: list[str] | None = None,
    search_all_experiments: bool = False,
    experiment_names: list[str] | None = None,
) -> Runs:
    """
    Search for Runs that fit the specified criteria.

    This function wraps the `mlflow.search_runs` function and returns the results
    as a `Runs` object. It allows for flexible searching of MLflow runs based on
    various criteria.

    Args:
        experiment_ids: List of experiment IDs. Search can work with experiment IDs or
            experiment names, but not both in the same call. Values other than
            ``None`` or ``[]`` will result in error if ``experiment_names`` is
            also not ``None`` or ``[]``. ``None`` will default to the active
            experiment if ``experiment_names`` is ``None`` or ``[]``.
        filter_string: Filter query string, defaults to searching all runs.
        run_view_type: one of enum values ``ACTIVE_ONLY``, ``DELETED_ONLY``, or ``ALL`` runs
            defined in :py:class:`mlflow.entities.ViewType`.
        max_results: The maximum number of runs to put in the dataframe. Default is 100,000
            to avoid causing out-of-memory issues on the user's machine.
        order_by: List of columns to order by (e.g., "metrics.rmse"). The ``order_by`` column
            can contain an optional ``DESC`` or ``ASC`` value. The default is ``ASC``.
            The default ordering is to sort by ``start_time DESC``, then ``run_id``.
        output_format: The output format to be returned. If ``pandas``, a ``pandas.DataFrame``
            is returned and, if ``list``, a list of :py:class:`mlflow.entities.Run`
            is returned.
        search_all_experiments: Boolean specifying whether all experiments should be searched.
            Only honored if ``experiment_ids`` is ``[]`` or ``None``.
        experiment_names: List of experiment names. Search can work with experiment IDs or
            experiment names, but not both in the same call. Values other
            than ``None`` or ``[]`` will result in error if ``experiment_ids``
            is also not ``None`` or ``[]``. ``None`` will default to the active
            experiment if ``experiment_ids`` is ``None`` or ``[]``.

    Returns:
        A `Runs` object containing the search results.
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
    return Runs(runs)  # type: ignore


@dataclass
class Runs:
    """
    A class to represent a collection of MLflow runs.

    This class provides methods to interact with the runs, such as filtering,
    retrieving specific runs, and accessing run information.
    """

    runs: list[Run]
    """A list of MLflow Run objects."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"

    def __len__(self) -> int:
        return len(self.runs)

    def filter(self, config: object) -> Runs:
        """
        Filter the runs based on the provided configuration.

        This method filters the runs in the collection according to the
        specified configuration object. The configuration object should
        contain key-value pairs that correspond to the parameters of the
        runs. Only the runs that match all the specified parameters will
        be included in the returned `Runs` object.

        Args:
            config: The configuration object to filter the runs.

        Returns:
            A new `Runs` object containing the filtered runs.
        """
        return Runs(filter_runs(self.runs, config))

    def get(self, config: object) -> Run | None:
        """
        Retrieve a specific run based on the provided configuration.

        This method filters the runs in the collection according to the
        specified configuration object and returns the run that matches
        the provided parameters. If more than one run matches the criteria,
        a `ValueError` is raised.

        Args:
            config: The configuration object to identify the run.

        Returns:
            Run: The run object that matches the provided configuration.
            None, if the runs are not in a DataFrame format.

        Raises:
            ValueError: If the number of filtered runs is not exactly one.
        """
        return get_run(self.runs, config)

    def get_earliest_run(self, config: object | None = None, **kwargs) -> Run | None:
        """
        Get the earliest run from the list of runs based on the start time.

        This method filters the runs based on the configuration if provided
        and returns the run with the earliest start time.

        Args:
            config: The configuration object to filter the runs.
                If None, no filtering is applied.
            **kwargs: Additional key-value pairs to filter the runs.

        Returns:
            The run with the earliest start time, or None if no runs match the criteria.
        """
        return get_earliest_run(self.runs, config, **kwargs)

    def get_latest_run(self, config: object | None = None, **kwargs) -> Run | None:
        """
        Get the latest run from the list of runs based on the start time.

        Args:
            config: The configuration object to filter the runs.
                If None, no filtering is applied.
            **kwargs: Additional key-value pairs to filter the runs.

        Returns:
            The run with the latest start time, or None if no runs match the criteria.
        """
        return get_latest_run(self.runs, config, **kwargs)

    def get_param_names(self) -> list[str]:
        """
        Get the parameter names from the runs.

        This method extracts the unique parameter names from the provided list of runs.
        It iterates through each run and collects the parameter names into a set to
        ensure uniqueness.

        Returns:
            A list of unique parameter names.
        """
        return get_param_names(self.runs)

    def get_param_dict(self) -> dict[str, list[str]]:
        """
        Get the parameter dictionary from the list of runs.

        This method extracts the parameter names and their corresponding values
        from the provided list of runs. It iterates through each run and collects
        the parameter values into a dictionary where the keys are parameter names
        and the values are lists of parameter values.

        Returns:
            A dictionary where the keys are parameter names and the values are lists
            of parameter values.
        """
        return get_param_dict(self.runs)


def filter_runs(runs: list[Run], config: object, **kwargs) -> list[Run]:
    """
    Filter the runs based on the provided configuration.

    This method filters the runs in the collection according to the
    specified configuration object. The configuration object should
    contain key-value pairs that correspond to the parameters of the
    runs. Only the runs that match all the specified parameters will
    be included in the returned list of runs.

    Args:
        runs: The runs to filter.
        config: The configuration object to filter the runs.
        **kwargs: Additional key-value pairs to filter the runs.

    Returns:
        A filtered list of runs.
    """
    for key, value in chain(iter_params(config), kwargs.items()):
        runs = [run for run in runs if _is_equal(run, key, value)]

        if len(runs) == 0:
            return []

    return runs


def _is_equal(run: Run, key: str, value: Any) -> bool:
    param = run.data.params.get(key, value)

    if param is None:
        return False

    return type(value)(param) == value


def get_run(runs: list[Run], config: object, **kwargs) -> Run | None:
    """
    Retrieve a specific run based on the provided configuration.

    This method filters the runs in the collection according to the
    specified configuration object and returns the run that matches
    the provided parameters. If more than one run matches the criteria,
    a `ValueError` is raised.

    Args:
        runs: The runs to filter.
        config: The configuration object to identify the run.
        **kwargs: Additional key-value pairs to filter the runs.

    Returns:
        The run object that matches the provided configuration, or None
        if no runs match the criteria.

    Raises:
        ValueError: If more than one run matches the criteria.
    """
    runs = filter_runs(runs, config, **kwargs)

    if len(runs) == 0:
        return None

    if len(runs) == 1:
        return runs[0]

    msg = f"Multiple runs were filtered. Expected number of runs is 1, but found {len(runs)} runs."
    raise ValueError(msg)


def get_earliest_run(runs: list[Run], config: object | None = None, **kwargs) -> Run | None:
    """
    Get the earliest run from the list of runs based on the start time.

    This method filters the runs based on the configuration if provided
    and returns the run with the earliest start time.

    Args:
        runs: The list of runs.
        config: The configuration object to filter the runs.
            If None, no filtering is applied.
        **kwargs: Additional key-value pairs to filter the runs.

    Returns:
        The run with the earliest start time, or None if no runs match the criteria.
    """
    if config is not None or kwargs:
        runs = filter_runs(runs, config or {}, **kwargs)

    return min(runs, key=lambda run: run.info.start_time, default=None)


def get_latest_run(runs: list[Run], config: object | None = None, **kwargs) -> Run | None:
    """
    Get the latest run from the list of runs based on the start time.

    This method filters the runs based on the configuration if provided
    and returns the run with the latest start time.

    Args:
        runs: The list of runs.
        config: The configuration object to filter the runs.
            If None, no filtering is applied.
        **kwargs: Additional key-value pairs to filter the runs.

    Returns:
        The run with the latest start time, or None if no runs match the criteria.
    """
    if config is not None or kwargs:
        runs = filter_runs(runs, config or {}, **kwargs)

    return max(runs, key=lambda run: run.info.start_time, default=None)


def get_param_names(runs: list[Run]) -> list[str]:
    """
    Get the parameter names from the runs.

    This method extracts the unique parameter names from the provided list of runs.
    It iterates through each run and collects the parameter names into a set to
    ensure uniqueness.

    Args:
        runs: The list of runs from which to extract parameter names.

    Returns:
        A list of unique parameter names.
    """
    param_names = set()

    for run in runs:
        for param in run.data.params.keys():
            param_names.add(param)

    return list(param_names)


def get_param_dict(runs: list[Run]) -> dict[str, list[str]]:
    """
    Get the parameter dictionary from the list of runs.

    This method extracts the parameter names and their corresponding values
    from the provided list of runs. It iterates through each run and collects
    the parameter values into a dictionary where the keys are parameter names
    and the values are lists of parameter values.

    Args:
        runs: The list of runs from which to extract parameter names and values.

    Returns:
        A dictionary where the keys are parameter names and the values are lists
        of parameter values.
    """
    params = {}

    for name in get_param_names(runs):
        it = (run.data.params[name] for run in runs if name in run.data.params)
        params[name] = sorted(set(it))

    return params


def load_config(run: Run) -> DictConfig:
    """
    Load the configuration for a given run.

    This function loads the configuration for the provided Run instance
    by downloading the configuration file from the MLflow artifacts and
    loading it using OmegaConf. It returns an empty config if
    `.hydra/config.yaml` is not found in the run's artifact directory.

    Args:
        run: The Run instance for which to load the configuration.

    Returns:
        The loaded configuration as a DictConfig object. Returns an empty
        DictConfig if the configuration file is not found.
    """
    run_id = run.info.run_id
    return _load_config(run_id)


@cache
def _load_config(run_id: str) -> DictConfig:
    try:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=".hydra/config.yaml",
        )
    except OSError:
        return DictConfig({})

    return OmegaConf.load(path)  # type: ignore


# def get_hydra_output_dir(run: Run_ | Series | str) -> Path:
#     """
#     Get the Hydra output directory.

#     Args:
#         run: The run object.

#     Returns:
#         Path: The Hydra output directory.
#     """
#     path = get_artifact_dir(run) / ".hydra/hydra.yaml"

#     if path.exists():
#         hc = OmegaConf.load(path)
#         return Path(hc.hydra.runtime.output_dir)

#     raise FileNotFoundError


# def log_hydra_output_dir(run: Run_ | Series | str) -> None:
#     """
#     Log the Hydra output directory.

#     Args:
#         run: The run object.

#     Returns:
#         None
#     """
#     output_dir = get_hydra_output_dir(run)
#     run_id = run if isinstance(run, str) else run.info.run_id
#     mlflow.log_artifacts(output_dir.as_posix(), run_id=run_id)
