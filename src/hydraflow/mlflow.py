"""
This module provides functionality to log parameters from Hydra
configuration objects and set up experiments using MLflow.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
from hydra.core.hydra_config import HydraConfig

from hydraflow.config import iter_params


def set_experiment(prefix: str = "", suffix: str = "", uri: str | None = None) -> None:
    """
    Set the experiment name and tracking URI optionally.

    This function sets the experiment name by combining the given prefix,
    the job name from HydraConfig, and the given suffix. Optionally, it can
    also set the tracking URI.

    Args:
        prefix: The prefix to prepend to the experiment name.
        suffix: The suffix to append to the experiment name.
        uri: The tracking URI to use.
    """
    if uri:
        mlflow.set_tracking_uri(uri)

    hc = HydraConfig.get()
    name = f"{prefix}{hc.job.name}{suffix}"
    mlflow.set_experiment(name)


def log_params(config: object, *, synchronous: bool | None = None) -> None:
    """
    Log the parameters from the given configuration object.

    This method logs the parameters from the provided configuration object
    using MLflow. It iterates over the parameters and logs them using the
    `mlflow.log_param` method.

    Args:
        config: The configuration object to log the parameters from.
        synchronous: Whether to log the parameters synchronously.
            Defaults to None.
    """
    for key, value in iter_params(config):
        mlflow.log_param(key, value, synchronous=synchronous)


def get_artifact_dir(artifact_path: str | None = None) -> Path:
    """
    Get the artifact directory for the given artifact path.

    This function retrieves the artifact URI for the specified artifact path
    using MLflow, downloads the artifacts to a local directory, and returns
    the path to that directory.

    Args:
        artifact_path: The artifact path for which to get the directory.
            Defaults to None.

    Returns:
        The local path to the directory where the artifacts are downloaded.
    """
    uri = mlflow.get_artifact_uri(artifact_path)
    dir = mlflow.artifacts.download_artifacts(artifact_uri=uri)

    return Path(dir)
