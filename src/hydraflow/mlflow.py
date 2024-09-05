"""
This module provides functionality to log parameters from Hydra
configuration objects and set up experiments using MLflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
from hydra.core.hydra_config import HydraConfig
from mlflow.tracking import artifact_utils
from omegaconf import OmegaConf

from hydraflow.config import iter_params

if TYPE_CHECKING:
    from mlflow.entities.experiment import Experiment
    from mlflow.entities.run import Run


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


def get_artifact_dir(run: Run | None = None) -> Path:
    """
    Retrieve the artifact directory for the given run.

    This function uses MLflow to get the artifact directory for the given run.

    Args:
        run (Run | None): The run object. Defaults to None.

    Returns:
        The local path to the directory where the artifacts are downloaded.
    """
    if run is None:
        uri = mlflow.get_artifact_uri()
    else:
        uri = artifact_utils.get_artifact_uri(run.info.run_id)

    return Path(mlflow.artifacts.download_artifacts(uri))


def get_hydra_output_dir(*, run: Run | None = None) -> Path:
    """
    Retrieve the Hydra output directory for the given run.

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
