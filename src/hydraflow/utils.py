"""Provide utility functions for HydraFlow."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import mlflow.artifacts
from hydra.core.hydra_config import HydraConfig
from mlflow.entities import Run
from mlflow.tracking import artifact_utils
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from mlflow.entities import Run


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
        uri = artifact_utils.get_artifact_uri(run.info.run_id)

    return Path(mlflow.artifacts.download_artifacts(uri))


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


def get_overrides() -> list[str]:
    """Retrieve the overrides for the current run."""
    return list(HydraConfig.get().overrides.task)  # ListConifg -> list


def load_overrides(run: Run) -> list[str]:
    """Load the overrides for a given run.

    This function loads the overrides for the provided Run instance
    by downloading the overrides file from the MLflow artifacts and
    loading it using OmegaConf. It returns an empty config if
    `.hydra/overrides.yaml` is not found in the run's artifact directory.

    Args:
        run (Run): The Run instance for which to load the overrides.

    Returns:
        The loaded overrides as a list of strings. Returns an empty list
        if the overrides file is not found.

    """
    path = get_artifact_dir(run) / ".hydra/overrides.yaml"
    return [str(x) for x in OmegaConf.load(path)]
