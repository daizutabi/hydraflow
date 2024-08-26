"""
This module provides functionality for managing and interacting with MLflow runs.
It includes classes and functions to filter runs, retrieve run information, and
log artifacts and configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
from mlflow.entities.run import Run as Run_
from mlflow.tracking import artifact_utils
from omegaconf import DictConfig, OmegaConf
from pandas import DataFrame, Series

from hydraflow.config import iter_params
from hydraflow.util import uri_to_path

if TYPE_CHECKING:
    from typing import Any


@dataclass
class Runs:
    """
    A class to represent a collection of MLflow runs.

    This class provides methods to interact with the runs, such as filtering,
    retrieving specific runs, and accessing run information.
    """

    runs: list[Run_] | DataFrame

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
            config (object): The configuration object to filter the runs.
                This object should contain key-value pairs representing
                the parameters to filter by.

        Returns:
            Runs: A new `Runs` object containing the filtered runs.
        """
        return Runs(filter_runs(self.runs, config))

    def get(self, config: object) -> Run:
        """
        Retrieve a specific run based on the provided configuration.

        This method filters the runs in the collection according to the
        specified configuration object and returns the run that matches
        the provided parameters. If more than one run matches the criteria,
        an error is raised.

        Args:
            config (object): The configuration object to identify the run.

        Returns:
            Run: The run object that matches the provided configuration.

        Raises:
            ValueError: If the number of filtered runs is not exactly one.
        """
        return Run(get_run(self.runs, config))

    def drop_unique_params(self) -> Runs:
        """
        Drop unique parameters from the runs and return a new Runs object.

        This method removes parameters that have unique values across all runs
        in the collection. This is useful for identifying common parameters
        that are shared among multiple runs.

        Returns:
            Runs: A new `Runs` object with unique parameters dropped.

        Raises:
            NotImplementedError: If the runs are not in a DataFrame format.
        """
        if isinstance(self.runs, DataFrame):
            return Runs(drop_unique_params(self.runs))

        raise NotImplementedError

    def get_param_names(self) -> list[str]:
        """
        Get the parameter names from the runs.

        This method extracts the parameter names from the runs in the collection.
        If the runs are stored in a DataFrame, it retrieves the column names
        that correspond to the parameters.

        Returns:
            list[str]: A list of parameter names.

        Raises:
            NotImplementedError: If the runs are not in a DataFrame format.
        """
        if isinstance(self.runs, DataFrame):
            return get_param_names(self.runs)

        raise NotImplementedError

    def get_param_dict(self) -> dict[str, list[str]]:
        """
        Get the parameter dictionary from the runs.

        This method extracts the parameter names and their corresponding values
        from the runs in the collection. If the runs are stored in a DataFrame,
        it retrieves the unique values for each parameter.


        Returns:
            dict[str, list[str]]: A dictionary of parameter names and their
            corresponding values.

        Raises:
            NotImplementedError: If the runs are not in a DataFrame format.
        """
        if isinstance(self.runs, DataFrame):
            return get_param_dict(self.runs)

        raise NotImplementedError


def search_runs(*args, **kwargs) -> Runs:
    """
    Search for runs that match the specified criteria.

    This function wraps the `mlflow.search_runs` function and returns the results
    as a `Runs` object.  It allows for flexible searching of MLflow runs based on
    various criteria.

    Args:
        *args: Positional arguments to pass to `mlflow.search_runs`.
        **kwargs: Keyword arguments to pass to `mlflow.search_runs`.

    Returns:
        Runs: A `Runs` object containing the search results.
    """
    runs = mlflow.search_runs(*args, **kwargs)
    return Runs(runs)


def filter_runs(runs: list[Run_] | DataFrame, config: object) -> list[Run_] | DataFrame:
    """
    Filter the runs based on the provided configuration.

    This method filters the runs in the collection according to the
    specified configuration object. The configuration object should
    contain key-value pairs that correspond to the parameters of the
    runs. Only the runs that match all the specified parameters will
    be included in the returned `Runs` object.

    Args:
        runs: The runs to filter.
        config: The configuration object to filter the runs.

    Returns:
        Runs: A filtered list of runs or a DataFrame.
    """
    if isinstance(runs, list):
        return _filter_runs_list(runs, config)

    return _filter_runs_dataframe(runs, config)


def _is_equal(run: Run_, key: str, value: Any) -> bool:
    param = run.data.params.get(key, value)

    if param is None:
        return False

    return type(value)(param) == value


def _filter_runs_list(runs: list[Run_], config: object) -> list[Run_]:
    for key, value in iter_params(config):
        runs = [run for run in runs if _is_equal(run, key, value)]

    return runs


def _filter_runs_dataframe(runs: DataFrame, config: object) -> DataFrame:
    index = np.ones(len(runs), dtype=bool)

    for key, value in iter_params(config):
        name = f"params.{key}"

        if name in runs:
            series = runs[name]
            is_value = -series.isna()
            param = series.fillna(value).astype(type(value))
            index &= is_value & (param == value)

    return runs[index]


def get_run(runs: list[Run_] | DataFrame, config: object) -> Run_ | Series:
    """
    Retrieve a specific run based on the provided configuration.

    This method filters the runs in the collection according to the
    specified configuration object and returns the run that matches
    the provided parameters. If more than one run matches the criteria,
    an error is raised.

    Args:
        runs: The runs to filter.
        config: The configuration object to identify the run.

    Returns:
        Run: The run object that matches the provided configuration.
    """
    runs = filter_runs(runs, config)

    if len(runs) == 1:
        return runs[0] if isinstance(runs, list) else runs.iloc[0]

    msg = f"number of filtered runs is not 1: got {len(runs)}"
    raise ValueError(msg)


def drop_unique_params(runs: DataFrame) -> DataFrame:
    """
    Drop unique parameters from the runs and return a new DataFrame.

    This method removes parameters that have unique values across all runs
    in the collection. This is useful for identifying common parameters
    that are shared among multiple runs.

    Args:
        runs: The DataFrame containing the runs.

    Returns:
        DataFrame: A new DataFrame with unique parameters dropped.
    """

    def select(column: str) -> bool:
        return not column.startswith("params.") or len(runs[column].unique()) > 1

    columns = [select(column) for column in runs.columns]
    return runs.iloc[:, columns]


def get_param_names(runs: DataFrame) -> list[str]:
    """
    Get the parameter names from the runs.

    This method extracts the parameter names from the runs in the collection.
    If the runs are stored in a DataFrame, it retrieves the column names
    that correspond to the parameters.

    Args:
        runs: The DataFrame containing the runs.

    Returns:
        list[str]: A list of parameter names.
    """

    def get_name(column: str) -> str:
        if column.startswith("params."):
            return column.split(".", maxsplit=1)[-1]

        return ""

    columns = [get_name(column) for column in runs.columns]
    return [column for column in columns if column]


def get_param_dict(runs: DataFrame) -> dict[str, list[str]]:
    """
    Get the parameter dictionary from the runs.

    This method extracts the parameter names and their corresponding values
    from the runs in the collection. If the runs are stored in a DataFrame,
    it retrieves the unique values for each parameter.

    Args:
        runs: The DataFrame containing the runs.

    Returns:
        dict[str, list[str]]: A dictionary of parameter names and
        their corresponding values.
    """
    params = {}
    for name in get_param_names(runs):
        params[name] = list(runs[f"params.{name}"].unique())

    return params


@dataclass
class Run:
    """
    A class to represent a specific MLflow run.

    This class provides methods to interact with the run, such as retrieving
    the run ID, artifact URI, and configuration. It also includes properties
    to access the artifact directory, artifact path, and Hydra output directory.
    """

    run: Run_ | Series | str

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.run_id!r})"

    @property
    def run_id(self) -> str:
        """
        Get the run ID.

        Returns:
            str: The run ID.
        """
        return get_run_id(self.run)

    def artifact_uri(self, artifact_path: str | None = None) -> str:
        """
        Get the artifact URI.

        Args:
            artifact_path (str | None): The artifact path.

        Returns:
            str: The artifact URI.
        """
        return get_artifact_uri(self.run, artifact_path)

    @property
    def artifact_dir(self) -> Path:
        """
        Get the artifact directory.

        Returns:
            Path: The artifact directory.
        """
        return get_artifact_dir(self.run)

    def artifact_path(self, artifact_path: str | None = None) -> Path:
        """
        Get the artifact path.

        Args:
            artifact_path: The artifact path.

        Returns:
            Path: The artifact path.
        """
        return get_artifact_path(self.run, artifact_path)

    @property
    def config(self) -> DictConfig:
        """
        Get the configuration.

        Returns:
            DictConfig: The configuration.
        """
        return load_config(self.run)

    def log_hydra_output_dir(self) -> None:
        """
        Log the Hydra output directory.

        Returns:
            None
        """
        log_hydra_output_dir(self.run)


def get_run_id(run: Run_ | Series | str) -> str:
    """
    Get the run ID.

    Args:
        run: The run object.

    Returns:
        str: The run ID.
    """
    if isinstance(run, str):
        return run

    if isinstance(run, Run_):
        return run.info.run_id

    return run.run_id


def get_artifact_uri(run: Run_ | Series | str, artifact_path: str | None = None) -> str:
    """
    Get the artifact URI.

    Args:
        run: The run object.
        artifact_path: The artifact path.

    Returns:
        str: The artifact URI.
    """
    run_id = get_run_id(run)
    return artifact_utils.get_artifact_uri(run_id, artifact_path)


def get_artifact_dir(run: Run_ | Series | str) -> Path:
    """
    Get the artifact directory.

    Args:
        run: The run object.

    Returns:
        Path: The artifact directory.
    """
    uri = get_artifact_uri(run)
    return uri_to_path(uri)


def get_artifact_path(run: Run_ | Series | str, artifact_path: str | None = None) -> Path:
    """
    Get the artifact path.

    Args:
        run: The run object.
        artifact_path: The artifact path.

    Returns:
        Path: The artifact path.
    """
    artifact_dir = get_artifact_dir(run)
    return artifact_dir / artifact_path if artifact_path else artifact_dir


def load_config(run: Run_ | Series | str) -> DictConfig:
    """
    Load the configuration.

    Args:
        run: The run object.

    Returns:
        DictConfig: The configuration.
    """
    run_id = get_run_id(run)
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


def get_hydra_output_dir(run: Run_ | Series | str) -> Path:
    """
    Get the Hydra output directory.

    Args:
        run: The run object.

    Returns:
        Path: The Hydra output directory.
    """
    path = get_artifact_dir(run) / ".hydra/hydra.yaml"

    if path.exists():
        hc = OmegaConf.load(path)
        return Path(hc.hydra.runtime.output_dir)

    raise FileNotFoundError


def log_hydra_output_dir(run: Run_ | Series | str) -> None:
    """
    Log the Hydra output directory.

    Args:
        run: The run object.

    Returns:
        None
    """
    output_dir = get_hydra_output_dir(run)
    run_id = run if isinstance(run, str) else run.info.run_id
    mlflow.log_artifacts(output_dir.as_posix(), run_id=run_id)
