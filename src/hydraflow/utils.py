"""Provide utility functions for HydraFlow."""

from __future__ import annotations

import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import mlflow.artifacts
from hydra.core.hydra_config import HydraConfig
from mlflow.entities import Run
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from collections.abc import Iterable


def get_artifact_dir(run: Run | None = None) -> Path:
    """Retrieve the artifact directory for the given run.

    This function uses MLflow to get the artifact directory for the given run.

    Args:
        run (Run | None): The run object. Defaults to None.

    Returns:
        The local path to the directory where the artifacts are downloaded.

    """
    if run is None:
        uri = mlflow.get_artifact_uri()
    else:
        uri = run.info.artifact_uri

    if not isinstance(uri, str):
        raise NotImplementedError

    return file_uri_to_path(uri)


def file_uri_to_path(uri: str) -> Path:
    """Convert a file URI to a local path."""
    if not uri.startswith("file:"):
        return Path(uri)

    path = urllib.parse.urlparse(uri).path
    return Path(urllib.request.url2pathname(path))  # for Windows


def get_artifact_path(run: Run | None, path: str) -> Path:
    """Retrieve the artifact path for the given run and path.

    This function uses MLflow to get the artifact path for the given run and path.

    Args:
        run (Run | None): The run object. Defaults to None.
        path (str): The path to the artifact.

    Returns:
        The local path to the artifact.

    """
    return get_artifact_dir(run) / path


def get_hydra_output_dir(run: Run | None = None) -> Path:
    """Retrieve the Hydra output directory for the given run.

    This function returns the Hydra output directory. If no run is provided,
    it retrieves the output directory from the current Hydra configuration.
    If a run is provided, it retrieves the artifact path for the run, loads
    the Hydra configuration from the downloaded artifacts, and returns the
    output directory specified in that configuration.

    Args:
        run (Run | None): The run object. Defaults to None.

    Returns:
        Path: The path to the Hydra output directory.

    Raises:
        FileNotFoundError: If the Hydra configuration file is not found
            in the artifacts.

    """
    if run is None:
        hc = HydraConfig.get()
        return Path(hc.runtime.output_dir)

    path = get_artifact_dir(run) / ".hydra/hydra.yaml"

    if path.exists():
        hc = OmegaConf.load(path)
        return Path(hc.hydra.runtime.output_dir)

    raise FileNotFoundError


def load_config(run: Run) -> DictConfig:
    """Load the configuration for a given run.

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
    path = get_artifact_dir(run) / ".hydra/config.yaml"
    return OmegaConf.load(path)  # type: ignore


def remove_run(run: Run | Iterable[Run]) -> None:
    """Remove the given run from the MLflow tracking server."""
    if not isinstance(run, Run):
        for r in run:
            remove_run(r)
        return

    shutil.rmtree(get_artifact_dir(run).parent)
