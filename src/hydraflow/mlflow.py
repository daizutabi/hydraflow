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


def get_artifact_dir(
    artifact_path: str | None = None,
    *,
    run_id: str | None = None,
) -> Path:
    """
    Get the artifact directory for the given artifact path.

    This function retrieves the artifact URI for the specified artifact path
    using MLflow, downloads the artifacts to a local directory, and returns
    the path to that directory.

    Args:
        artifact_path (str | None): The artifact path for which to get the
            directory. Defaults to None.
        run_id (str | None): The run ID for which to get the artifact directory.

    Returns:
        The local path to the directory where the artifacts are downloaded.
    """
    if run_id is None:
        uri = mlflow.get_artifact_uri(artifact_path)
    else:
        uri = artifact_utils.get_artifact_uri(run_id, artifact_path)

    dir = mlflow.artifacts.download_artifacts(artifact_uri=uri)

    return Path(dir)


def get_hydra_output_dir(*, run_id: str | None = None) -> Path:
    if run_id is None:
        hc = HydraConfig.get()
        return Path(hc.runtime.output_dir)

    path = get_artifact_dir(run_id=run_id) / ".hydra/hydra.yaml"

    if path.exists():
        hc = OmegaConf.load(path)
        return Path(hc.hydra.runtime.output_dir)

    raise FileNotFoundError


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
