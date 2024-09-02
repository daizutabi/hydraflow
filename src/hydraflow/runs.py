"""
This module provides functionality for managing and interacting with MLflow
runs. It includes the `RunCollection` class and various methods to filter
runs, retrieve run information, log artifacts, and load configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from itertools import chain
from typing import TYPE_CHECKING, Any, TypeVar

import mlflow
from mlflow.artifacts import download_artifacts
from mlflow.entities import ViewType
from mlflow.entities.run import Run
from mlflow.tracking.fluent import SEARCH_MAX_RESULTS_PANDAS
from omegaconf import DictConfig, OmegaConf

from hydraflow.config import iter_params

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Any


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
        experiment_ids (list[str] | None): List of experiment IDs. Search can
            work with experiment IDs or experiment names, but not both in the
            same call. Values other than ``None`` or ``[]`` will result in
            error if ``experiment_names`` is also not ``None`` or ``[]``.
            ``experiment_names`` is also not ``None`` or ``[]``. ``None`` will
            default to the active experiment if ``experiment_names`` is ``None``
            or ``[]``.
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


T = TypeVar("T")


@dataclass
class RunCollection:
    """
    A class to represent a collection of MLflow runs.

    This class provides methods to interact with the runs, such as filtering,
    retrieving specific runs, and accessing run information.
    """

    _runs: list[Run]
    """A list of MLflow Run objects."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"

    def __len__(self) -> int:
        return len(self._runs)

    def __iter__(self) -> Iterator[Run]:
        return iter(self._runs)

    def __getitem__(self, index: int) -> Run:
        return self._runs[index]

    def __contains__(self, run: Run) -> bool:
        return run in self._runs

    def sort(
        self,
        key: Callable[[Run], Any] | None = None,
        reverse: bool = False,
    ) -> None:
        self._runs.sort(key=key or (lambda x: x.info.start_time), reverse=reverse)

    def first(self) -> Run:
        """
        Get the first run in the collection.

        Returns:
            The first run object in the collection.

        Raises:
            ValueError: If the collection is empty.
        """
        if not self._runs:
            raise ValueError("The collection is empty.")

        return self._runs[0]

    def try_first(self) -> Run | None:
        """
        Try to get the first run in the collection.

        Returns:
            The first run object in the collection, or None if the collection
            is empty.
        """
        return self._runs[0] if self._runs else None

    def last(self) -> Run:
        """
        Get the last run in the collection.

        Returns:
            The last run object in the collection.

        Raises:
            ValueError: If the collection is empty.
        """
        if not self._runs:
            raise ValueError("The collection is empty.")

        return self._runs[-1]

    def try_last(self) -> Run | None:
        """
        Try to get the last run in the collection.

        Returns:
            The last run object in the collection, or None if the collection is
            empty.
        """
        return self._runs[-1] if self._runs else None

    def filter(self, config: object | None = None, **kwargs) -> RunCollection:
        """
        Filter the runs based on the provided configuration.

        This method filters the runs in the collection according to the
        specified configuration object and additional key-value pairs. The
        configuration object and key-value pairs should contain key-value pairs
        that correspond to the parameters of the runs. Only the runs that match
        all the specified parameters will be included in the returned
        `RunCollection` object.

        The filtering supports:
        - Exact matches for single values.
        - Membership checks for lists of values.
        - Range checks for tuples of two values (inclusive of the lower bound
          and exclusive of the upper bound).

        Args:
            config (object | None): The configuration object to filter the runs.
                This can be any object that provides key-value pairs through
                the `iter_params` function.
            **kwargs: Additional key-value pairs to filter the runs.

        Returns:
            A new `RunCollection` object containing the filtered runs.
        """
        return RunCollection(filter_runs(self._runs, config, **kwargs))

    def find(self, config: object | None = None, **kwargs) -> Run:
        """
        Find the first run based on the provided configuration.

        This method filters the runs in the collection according to the
        specified configuration object and returns the first run that matches
        the provided parameters. If no run matches the criteria, a `ValueError`
        is raised.

        Args:
            config (object | None): The configuration object to identify the run.
            **kwargs: Additional key-value pairs to filter the runs.

        Returns:
            The first run object that matches the provided configuration.

        Raises:
            ValueError: If no run matches the criteria.

        See Also:
            RunCollection.filter: The method that performs the actual filtering
            logic.
        """
        return find_run(self._runs, config, **kwargs)

    def try_find(self, config: object | None = None, **kwargs) -> Run | None:
        """
        Find the first run based on the provided configuration.

        This method filters the runs in the collection according to the
        specified configuration object and returns the first run that matches
        the provided parameters. If no run matches the criteria, None is
        returned.

        Args:
            config (object | None): The configuration object to identify the run.
            **kwargs: Additional key-value pairs to filter the runs.

        Returns:
            The first run object that matches the provided configuration, or
            None if no runs match the criteria.

        See Also:
            RunCollection.filter: The method that performs the actual filtering
            logic.
        """
        return try_find_run(self._runs, config, **kwargs)

    def find_last(self, config: object | None = None, **kwargs) -> Run:
        """
        Find the last run based on the provided configuration.

        This method filters the runs in the collection according to the
        specified configuration object and returns the last run that matches
        the provided parameters. If no run matches the criteria, a `ValueError`
        is raised.

        Args:
            config (object | None): The configuration object to identify the run.
            **kwargs: Additional key-value pairs to filter the runs.

        Returns:
            The last run object that matches the provided configuration.

        Raises:
            ValueError: If no run matches the criteria.

        See Also:
            RunCollection.filter: The method that performs the actual filtering
            logic.
        """
        return find_last_run(self._runs, config, **kwargs)

    def try_find_last(self, config: object | None = None, **kwargs) -> Run | None:
        """
        Find the last run based on the provided configuration.

        This method filters the runs in the collection according to the
        specified configuration object and returns the last run that matches
        the provided parameters. If no run matches the criteria, None is
        returned.

        Args:
            config (object | None): The configuration object to identify the run.
            **kwargs: Additional key-value pairs to filter the runs.

        Returns:
            The last run object that matches the provided configuration, or
            None if no runs match the criteria.

        See Also:
            RunCollection.filter: The method that performs the actual filtering
            logic.
        """
        return try_find_last_run(self._runs, config, **kwargs)

    def get(self, config: object | None = None, **kwargs) -> Run:
        """
        Retrieve a specific run based on the provided configuration.

        This method filters the runs in the collection according to the
        specified configuration object and returns the run that matches the
        provided parameters. If no run matches the criteria, or if more than
        one run matches the criteria, a `ValueError` is raised.

        Args:
            config (object | None): The configuration object to identify the run.
            **kwargs: Additional key-value pairs to filter the runs.

        Returns:
            The run object that matches the provided configuration.

        Raises:
            ValueError: If no run matches the criteria or if more than one run
            matches the criteria.

        See Also:
            RunCollection.filter: The method that performs the actual filtering
            logic.
        """
        return get_run(self._runs, config, **kwargs)

    def try_get(self, config: object | None = None, **kwargs) -> Run | None:
        """
        Retrieve a specific run based on the provided configuration.

        This method filters the runs in the collection according to the
        specified configuration object and returns the run that matches the
        provided parameters. If no run matches the criteria, None is returned.
        If more than one run matches the criteria, a `ValueError` is raised.

        Args:
            config (object | None): The configuration object to identify the run.
            **kwargs: Additional key-value pairs to filter the runs.

        Returns:
            The run object that matches the provided configuration, or None if
            no runs match the criteria.

        Raises:
            ValueError: If more than one run matches the criteria.

        See Also:
            RunCollection.filter: The method that performs the actual filtering
            logic.
        """
        return try_get_run(self._runs, config, **kwargs)

    def get_param_names(self) -> list[str]:
        """
        Get the parameter names from the runs.

        This method extracts the unique parameter names from the provided list
        of runs. It iterates through each run and collects the parameter names
        into a set to ensure uniqueness.

        Returns:
            A list of unique parameter names.
        """
        return get_param_names(self._runs)

    def get_param_dict(self) -> dict[str, list[str]]:
        """
        Get the parameter dictionary from the list of runs.

        This method extracts the parameter names and their corresponding values
        from the provided list of runs. It iterates through each run and
        collects the parameter values into a dictionary where the keys are
        parameter names and the values are lists of parameter values.

        Returns:
            A dictionary where the keys are parameter names and the values are
            lists of parameter values.
        """
        return get_param_dict(self._runs)

    def map(self, func: Callable[[Run], T]) -> Iterator[T]:
        """
        Apply a function to each run in the collection and return an iterator of
        results.

        Args:
            func (Callable[[Run], T]): A function that takes a run and returns a
                result.

        Yields:
            Results obtained by applying the function to each run in the
            collection.
        """
        return (func(run) for run in self._runs)

    def map_run_id(self, func: Callable[[str], T]) -> Iterator[T]:
        """
        Apply a function to each run id in the collection and return an iterator
        of results.

        Args:
            func (Callable[[str], T]): A function that takes a run id and returns a
                result.

        Yields:
            Results obtained by applying the function to each run id in the
            collection.
        """
        return (func(run.info.run_id) for run in self._runs)

    def map_config(self, func: Callable[[DictConfig], T]) -> Iterator[T]:
        """
        Apply a function to each run configuration in the collection and return
        an iterator of results.

        Args:
            func (Callable[[DictConfig], T]): A function that takes a run
                configuration and returns a result.

        Yields:
            Results obtained by applying the function to each run configuration
            in the collection.
        """
        return (func(load_config(run)) for run in self._runs)

    def map_uri(self, func: Callable[[str | None], T]) -> Iterator[T]:
        """
        Apply a function to each artifact URI in the collection and return an
        iterator of results.

        This method iterates over each run in the collection, retrieves the
        artifact URI, and applies the provided function to it. If a run does not
        have an artifact URI, None is passed to the function.

        Args:
            func (Callable[[str | None], T]): A function that takes an
            artifact URI (string or None) and returns a result.

        Yields:
            Results obtained by applying the function to each artifact URI in the
            collection.
        """
        return (func(run.info.artifact_uri) for run in self._runs)

    def map_dir(self, func: Callable[[str], T]) -> Iterator[T]:
        """
        Apply a function to each artifact directory in the collection and return
        an iterator of results.

        This method iterates over each run in the collection, downloads the
        artifact directory, and applies the provided function to the directory
        path.

        Args:
            func (Callable[[str], T]): A function that takes an artifact directory
                path (string) and returns a result.

        Yields:
            Results obtained by applying the function to each artifact directory
            in the collection.
        """
        return (func(download_artifacts(run_id=run.info.run_id)) for run in self._runs)


def _param_matches(run: Run, key: str, value: Any) -> bool:
    """
    Check if the run's parameter matches the specified key-value pair.

    This function checks if the run's parameters contain the specified
    key-value pair. It handles different types of values, including lists
    and tuples.

    Args:
        run (Run): The run object to check.
        key (str): The parameter key to check.
        value (Any): The parameter value to check.

    Returns:
        True if the run's parameter matches the specified key-value pair,
        False otherwise.
    """
    param = run.data.params.get(key, value)

    if param is None:
        return False

    if param == "None":
        return value is None

    if isinstance(value, list) and value:
        return type(value[0])(param) in value

    if isinstance(value, tuple) and len(value) == 2:
        return value[0] <= type(value[0])(param) < value[1]

    return type(value)(param) == value


def filter_runs(runs: list[Run], config: object | None = None, **kwargs) -> list[Run]:
    """
    Filter the runs based on the provided configuration.

    This method filters the runs in the collection according to the
    specified configuration object and additional key-value pairs.
    The configuration object and key-value pairs should contain
    key-value pairs that correspond to the parameters of the runs.
    Only the runs that match all the specified parameters will
    be included in the returned list of runs.

    The filtering supports:
    - Exact matches for single values.
    - Membership checks for lists of values.
    - Range checks for tuples of two values (inclusive of the lower bound and
      exclusive of the upper bound).

    Args:
        runs (list[Run]): The list of runs to filter.
        config (object | None): The configuration object to filter the runs.
            This can be any object that provides key-value pairs through the
            `iter_params` function.
        **kwargs: Additional key-value pairs to filter the runs.

    Returns:
        A list of runs that match the specified configuration and key-value pairs.
    """
    for key, value in chain(iter_params(config), kwargs.items()):
        runs = [run for run in runs if _param_matches(run, key, value)]

        if len(runs) == 0:
            return []

    return runs


def find_run(runs: list[Run], config: object | None = None, **kwargs) -> Run:
    """
    Find the first run based on the provided configuration.

    This method filters the runs in the collection according to the
    specified configuration object and returns the first run that matches
    the provided parameters. If no run matches the criteria, a `ValueError` is
    raised.

    Args:
        runs (list[Run]): The runs to filter.
        config (object | None): The configuration object to identify the run.
        **kwargs: Additional key-value pairs to filter the runs.

    Returns:
        The first run object that matches the provided configuration.

    Raises:
        ValueError: If no run matches the criteria.

    See Also:
        RunCollection.filter: The method that performs the actual filtering logic.
    """
    filtered_runs = filter_runs(runs, config, **kwargs)

    if len(filtered_runs) == 0:
        raise ValueError("No run matches the provided configuration.")

    return filtered_runs[0]


def try_find_run(runs: list[Run], config: object | None = None, **kwargs) -> Run | None:
    """
    Find the first run based on the provided configuration.

    This method filters the runs in the collection according to the
    specified configuration object and returns the first run that matches
    the provided parameters. If no run matches the criteria, None is returned.

    Args:
        runs (list[Run]): The runs to filter.
        config (object | None): The configuration object to identify the run.
        **kwargs: Additional key-value pairs to filter the runs.

    Returns:
        The first run object that matches the provided configuration, or None
        if no runs match the criteria.
    """
    filtered_runs = filter_runs(runs, config, **kwargs)

    if len(filtered_runs) == 0:
        return None

    return filtered_runs[0]


def find_last_run(runs: list[Run], config: object | None = None, **kwargs) -> Run:
    """
    Find the last run based on the provided configuration.

    This method filters the runs in the collection according to the
    specified configuration object and returns the last run that matches
    the provided parameters. If no run matches the criteria, a `ValueError`
    is raised.

    Args:
        runs (list[Run]): The runs to filter.
        config (object | None): The configuration object to identify the run.
        **kwargs: Additional key-value pairs to filter the runs.

    Returns:
        The last run object that matches the provided configuration.

    Raises:
        ValueError: If no run matches the criteria.

    See Also:
        RunCollection.filter: The method that performs the actual filtering
        logic.
    """
    filtered_runs = filter_runs(runs, config, **kwargs)

    if len(filtered_runs) == 0:
        raise ValueError("No run matches the provided configuration.")

    return filtered_runs[-1]


def try_find_last_run(runs: list[Run], config: object | None = None, **kwargs) -> Run | None:
    """
    Find the last run based on the provided configuration.

    This method filters the runs in the collection according to the
    specified configuration object and returns the last run that matches
    the provided parameters. If no run matches the criteria, None is returned.

    Args:
        runs (list[Run]): The runs to filter.
        config (object | None): The configuration object to identify the run.
        **kwargs: Additional key-value pairs to filter the runs.

    Returns:
        The last run object that matches the provided configuration, or None
        if no runs match the criteria.
    """
    filtered_runs = filter_runs(runs, config, **kwargs)

    if len(filtered_runs) == 0:
        return None

    return filtered_runs[-1]


def get_run(runs: list[Run], config: object | None = None, **kwargs) -> Run:
    """
    Retrieve a specific run based on the provided configuration.

    This method filters the runs in the collection according to the
    specified configuration object and returns the run that matches
    the provided parameters. If no run matches the criteria, or if more
    than one run matches the criteria, a `ValueError` is raised.

    Args:
        runs (list[Run]): The runs to filter.
        config (object | None): The configuration object to identify the run.
        **kwargs: Additional key-value pairs to filter the runs.

    Returns:
        The run object that matches the provided configuration.

    Raises:
        ValueError: If no run matches the criteria or if more than one run
        matches the criteria.

    See Also:
        RunCollection.filter: The method that performs the actual filtering
        logic.
    """
    filtered_runs = filter_runs(runs, config, **kwargs)

    if len(filtered_runs) == 0:
        raise ValueError("No run matches the provided configuration.")

    if len(filtered_runs) == 1:
        return filtered_runs[0]

    msg = (
        f"Multiple runs were filtered. Expected number of runs is 1, "
        f"but found {len(filtered_runs)} runs."
    )
    raise ValueError(msg)


def try_get_run(runs: list[Run], config: object | None = None, **kwargs) -> Run | None:
    """
    Retrieve a specific run based on the provided configuration.

    This method filters the runs in the collection according to the
    specified configuration object and returns the run that matches
    the provided parameters. If no run matches the criteria, None is returned.
    If more than one run matches the criteria, a `ValueError` is raised.

    Args:
        runs (list[Run]): The runs to filter.
        config (object | None): The configuration object to identify the run.
        **kwargs: Additional key-value pairs to filter the runs.

    Returns:
        The run object that matches the provided configuration, or None
        if no runs match the criteria.

    Raises:
        ValueError: If more than one run matches the criteria.

    See Also:
        RunCollection.filter: The method that performs the actual filtering
        logic.
    """
    filtered_runs = filter_runs(runs, config, **kwargs)

    if len(filtered_runs) == 0:
        return None

    if len(filtered_runs) == 1:
        return filtered_runs[0]

    msg = (
        "Multiple runs were filtered. Expected number of runs is 1, "
        f"but found {len(filtered_runs)} runs."
    )
    raise ValueError(msg)


def get_param_names(runs: list[Run]) -> list[str]:
    """
    Get the parameter names from the runs.

    This method extracts the unique parameter names from the provided list of
    runs. It iterates through each run and collects the parameter names into a
    set to ensure uniqueness.

    Args:
        runs (list[Run]): The list of runs from which to extract parameter names.

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
        runs (list[Run]): The list of runs from which to extract parameter names
        and values.

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
        run (Run): The Run instance for which to load the configuration.

    Returns:
        The loaded configuration as a DictConfig object. Returns an empty
        DictConfig if the configuration file is not found.
    """
    run_id = run.info.run_id
    return _load_config(run_id)


@cache
def _load_config(run_id: str) -> DictConfig:
    try:
        path = download_artifacts(run_id=run_id, artifact_path=".hydra/config.yaml")
    except OSError:
        return DictConfig({})

    return OmegaConf.load(path)  # type: ignore
