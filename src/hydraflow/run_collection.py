"""
Provide functionality for managing and interacting with MLflow runs.
It includes the `RunCollection` class, which serves as a container
for multiple MLflow `Run` instances, and various methods to filter and
retrieve these runs.

Key Features:
- **Run Management**: The `RunCollection` class allows for easy management of
  multiple MLflow runs, providing methods to filter and retrieve runs based
  on various criteria.
- **Filtering**: Support filtering runs based on specific configurations
  and parameters, enabling users to easily find runs that match certain conditions.
- **Retrieval**: Users can retrieve specific runs, including the first, last, or any
  run that matches a given configuration.
- **Artifact Handling**: Provide methods to access and manipulate the
  artifacts associated with each run, including retrieving artifact URIs and
  directories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

from mlflow.entities.run import Run

from hydraflow.config import iter_params
from hydraflow.info import RunCollectionInfo

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from typing import Any

    from omegaconf import DictConfig


T = TypeVar("T")
P = ParamSpec("P")


@dataclass
class RunCollection:
    """
    Represent a collection of MLflow runs.

    Provide methods to interact with the runs, such as filtering,
    retrieving specific runs, and accessing run information.

    Key Features:
    - Filtering: Easily filter runs based on various criteria.
    - Retrieval: Access specific runs by index or through methods.
    - Metadata: Access run metadata and associated information.
    """

    _runs: list[Run]
    """A list of MLflow `Run` instances."""

    _info: RunCollectionInfo = field(init=False)
    """An instance of `RunCollectionInfo`."""

    def __post_init__(self):
        self._info = RunCollectionInfo(self)

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

    @property
    def info(self) -> RunCollectionInfo:
        """An instance of `RunCollectionInfo`."""
        return self._info

    def sort(
        self,
        key: Callable[[Run], Any] | None = None,
        reverse: bool = False,
    ) -> None:
        self._runs.sort(key=key or (lambda x: x.info.start_time), reverse=reverse)

    def one(self) -> Run:
        """
        Get the only `Run` instance in the collection.

        Returns:
            The only `Run` instance in the collection.

        Raises:
            ValueError: If the collection does not contain exactly one run.
        """
        if len(self._runs) != 1:
            raise ValueError("The collection does not contain exactly one run.")

        return self._runs[0]

    def try_one(self) -> Run | None:
        """
        Try to get the only `Run` instance in the collection.

        Returns:
            The only `Run` instance in the collection, or None if the collection
            does not contain exactly one run.
        """
        return self._runs[0] if len(self._runs) == 1 else None

    def first(self) -> Run:
        """
        Get the first `Run` instance in the collection.

        Returns:
            The first `Run` instance in the collection.

        Raises:
            ValueError: If the collection is empty.
        """
        if not self._runs:
            raise ValueError("The collection is empty.")

        return self._runs[0]

    def try_first(self) -> Run | None:
        """
        Try to get the first `Run` instance in the collection.

        Returns:
            The first `Run` instance in the collection, or None if the collection
            is empty.
        """
        return self._runs[0] if self._runs else None

    def last(self) -> Run:
        """
        Get the last `Run` instance in the collection.

        Returns:
            The last `Run` instance in the collection.

        Raises:
            ValueError: If the collection is empty.
        """
        if not self._runs:
            raise ValueError("The collection is empty.")

        return self._runs[-1]

    def try_last(self) -> Run | None:
        """
        Try to get the last `Run` instance in the collection.

        Returns:
            The last `Run` instance in the collection, or None if the collection
            is empty.
        """
        return self._runs[-1] if self._runs else None

    def filter(self, config: object | None = None, **kwargs) -> RunCollection:
        """
        Filter the `Run` instances based on the provided configuration.

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
        Find the first `Run` instance based on the provided configuration.

        This method filters the runs in the collection according to the
        specified configuration object and returns the first run that matches
        the provided parameters. If no run matches the criteria, a `ValueError`
        is raised.

        Args:
            config (object | None): The configuration object to identify the run.
            **kwargs: Additional key-value pairs to filter the runs.

        Returns:
            The first `Run` instance that matches the provided configuration.

        Raises:
            ValueError: If no run matches the criteria.

        See Also:
            `filter`: Perform the actual filtering logic.
        """
        try:
            return self.filter(config, **kwargs).first()
        except ValueError:
            raise ValueError("No run matches the provided configuration.")

    def try_find(self, config: object | None = None, **kwargs) -> Run | None:
        """
        Try to find the first `Run` instance based on the provided configuration.

        This method filters the runs in the collection according to the
        specified configuration object and returns the first run that matches
        the provided parameters. If no run matches the criteria, None is
        returned.

        Args:
            config (object | None): The configuration object to identify the run.
            **kwargs: Additional key-value pairs to filter the runs.

        Returns:
            The first `Run` instance that matches the provided configuration, or
            None if no runs match the criteria.

        See Also:
            `filter`: Perform the actual filtering logic.
        """
        return self.filter(config, **kwargs).try_first()

    def find_last(self, config: object | None = None, **kwargs) -> Run:
        """
        Find the last `Run` instance based on the provided configuration.

        This method filters the runs in the collection according to the
        specified configuration object and returns the last run that matches
        the provided parameters. If no run matches the criteria, a `ValueError`
        is raised.

        Args:
            config (object | None): The configuration object to identify the run.
            **kwargs: Additional key-value pairs to filter the runs.

        Returns:
            The last `Run` instance that matches the provided configuration.

        Raises:
            ValueError: If no run matches the criteria.

        See Also:
            `filter`: Perform the actual filtering logic.
        """
        try:
            return self.filter(config, **kwargs).last()
        except ValueError:
            raise ValueError("No run matches the provided configuration.")

    def try_find_last(self, config: object | None = None, **kwargs) -> Run | None:
        """
        Try to find the last `Run` instance based on the provided configuration.

        This method filters the runs in the collection according to the
        specified configuration object and returns the last run that matches
        the provided parameters. If no run matches the criteria, None is
        returned.

        Args:
            config (object | None): The configuration object to identify the run.
            **kwargs: Additional key-value pairs to filter the runs.

        Returns:
            The last `Run` instance that matches the provided configuration, or
            None if no runs match the criteria.

        See Also:
            `filter`: Perform the actual filtering logic.
        """
        return self.filter(config, **kwargs).try_last()

    def get(self, config: object | None = None, **kwargs) -> Run:
        """
        Retrieve a specific `Run` instance based on the provided configuration.

        This method filters the runs in the collection according to the
        specified configuration object and returns the run that matches the
        provided parameters. If no run matches the criteria, or if more than
        one run matches the criteria, a `ValueError` is raised.

        Args:
            config (object | None): The configuration object to identify the run.
            **kwargs: Additional key-value pairs to filter the runs.

        Returns:
            The `Run` instance that matches the provided configuration.

        Raises:
            ValueError: If no run matches the criteria or if more than one run
            matches the criteria.

        See Also:
            `filter`: Perform the actual filtering logic.
        """
        try:
            return self.filter(config, **kwargs).one()
        except ValueError:
            msg = "The filtered collection does not contain exactly one run."
            raise ValueError(msg)

    def try_get(self, config: object | None = None, **kwargs) -> Run | None:
        """
        Try to retrieve a specific `Run` instance based on the provided configuration.

        This method filters the runs in the collection according to the
        specified configuration object and returns the run that matches the
        provided parameters. If no run matches the criteria, None is returned.
        If more than one run matches the criteria, a `ValueError` is raised.

        Args:
            config (object | None): The configuration object to identify the run.
            **kwargs: Additional key-value pairs to filter the runs.

        Returns:
            The `Run` instance that matches the provided configuration, or None
            if no runs match the criteria.

        Raises:
            ValueError: If more than one run matches the criteria.

        See Also:
            `filter`: Perform the actual filtering logic.
        """
        return self.filter(config, **kwargs).try_one()

    def get_param_names(self) -> list[str]:
        """
        Get the parameter names from the runs.

        This method extracts the unique parameter names from the provided list
        of runs. It iterates through each run and collects the parameter names
        into a set to ensure uniqueness.

        Returns:
            A list of unique parameter names.
        """
        param_names = set()

        for run in self:
            for param in run.data.params.keys():
                param_names.add(param)

        return list(param_names)

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
        params = {}

        for name in self.get_param_names():
            it = (run.data.params[name] for run in self if name in run.data.params)
            params[name] = sorted(set(it))

        return params

    def map(
        self,
        func: Callable[Concatenate[Run, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterator[T]:
        """
        Apply a function to each run in the collection and return an iterator of
        results.

        This method iterates over each run in the collection and applies the
        provided function to it, along with any additional arguments and
        keyword arguments.

        Args:
            func (Callable[[Run, P], T]): A function that takes a run and
                additional arguments and returns a result.
            *args: Additional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.

        Yields:
            Results obtained by applying the function to each run in the collection.
        """
        return (func(run, *args, **kwargs) for run in self)

    def map_run_id(
        self,
        func: Callable[Concatenate[str, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterator[T]:
        """
        Apply a function to each run id in the collection and return an iterator
        of results.

        Args:
            func (Callable[[str, P], T]): A function that takes a run id and returns a
                result.
            *args: Additional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.

        Yields:
            Results obtained by applying the function to each run id in the
            collection.
        """
        return (func(run_id, *args, **kwargs) for run_id in self.info.run_id)

    def map_config(
        self,
        func: Callable[Concatenate[DictConfig, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterator[T]:
        """
        Apply a function to each run configuration in the collection and return
        an iterator of results.

        Args:
            func (Callable[[DictConfig, P], T]): A function that takes a run
                configuration and returns a result.
            *args: Additional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.

        Yields:
            Results obtained by applying the function to each run configuration
            in the collection.
        """
        return (func(config, *args, **kwargs) for config in self.info.config)

    def map_uri(
        self,
        func: Callable[Concatenate[str | None, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterator[T]:
        """
        Apply a function to each artifact URI in the collection and return an
        iterator of results.

        Iterate over each run in the collection, retrieves the artifact URI, and
        apply the provided function to it. If a run does not have an artifact
        URI, None is passed to the function.

        Args:
            func (Callable[[str | None, P], T]): A function that takes an
                artifact URI (string or None) and returns a result.
            *args: Additional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.

        Yields:
            Results obtained by applying the function to each artifact URI in the
            collection.
        """
        return (func(uri, *args, **kwargs) for uri in self.info.artifact_uri)

    def map_dir(
        self,
        func: Callable[Concatenate[Path, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterator[T]:
        """
        Apply a function to each artifact directory in the collection and return
        an iterator of results.

        Iterate over each run in the collection, downloads the artifact
        directory, and apply the provided function to the directory path.

        Args:
            func (Callable[[Path, P], T]): A function that takes an artifact directory
                path (string) and returns a result.
            *args: Additional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.

        Yields:
            Results obtained by applying the function to each artifact directory
            in the collection.
        """
        return (func(dir, *args, **kwargs) for dir in self.info.artifact_dir)

    def group_by(
        self, *names: str | list[str]
    ) -> dict[tuple[str | None, ...], RunCollection]:
        """
        Group runs by specified parameter names.

        Group the runs in the collection based on the values of the
        specified parameters. Each unique combination of parameter values will
        form a key in the returned dictionary.

        Args:
            *names (str | list[str]): The names of the parameters to group by.
                This can be a single parameter name or multiple names provided
                as separate arguments or as a list.

        Returns:
            dict[tuple[str | None, ...], RunCollection]: A dictionary where the keys
            are tuples of parameter values and the values are RunCollection objects
            containing the runs that match those parameter values.
        """
        grouped_runs: dict[tuple[str | None, ...], list[Run]] = {}
        for run in self._runs:
            key = get_params(run, *names)
            grouped_runs.setdefault(key, []).append(run)

        return {key: RunCollection(runs) for key, runs in grouped_runs.items()}


def _param_matches(run: Run, key: str, value: Any) -> bool:
    """
    Check if the run's parameter matches the specified key-value pair.

    Check if the run's parameters contain the specified
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

    Filter the runs in the collection according to the
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


def get_params(run: Run, *names: str | list[str]) -> tuple[str | None, ...]:
    """
    Retrieve the values of specified parameters from the given run.

    This function extracts the values of the parameters identified by the
    provided names from the specified run. It can accept both individual
    parameter names and lists of parameter names.

    Args:
        run (Run): The run object from which to extract parameter values.
        *names (str | list[str]): The names of the parameters to retrieve.
            This can be a single parameter name or multiple names provided
            as separate arguments or as a list.

    Returns:
        tuple[str | None, ...]: A tuple containing the values of the specified
        parameters in the order they were provided.
    """
    names_ = []
    for name in names:
        if isinstance(name, list):
            names_.extend(name)
        else:
            names_.append(name)

    return tuple(run.data.params.get(name) for name in names_)
