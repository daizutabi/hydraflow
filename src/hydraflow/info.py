"""Provide information about MLflow runs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
from hydra.core.hydra_config import HydraConfig
from mlflow.tracking import artifact_utils
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from mlflow.entities import Run

    from hydraflow.run_collection import RunCollection


class RunCollectionInfo:
    """Provide information about MLflow runs."""

    def __init__(self, runs: RunCollection) -> None:
        self._runs = runs

    @property
    def run_id(self) -> list[str]:
        """Get the run ID for each run in the collection."""
        return [run.info.run_id for run in self._runs]

    @property
    def params(self) -> list[dict[str, str]]:
        """Get the parameters for each run in the collection."""
        return [run.data.params for run in self._runs]

    @property
    def metrics(self) -> list[dict[str, float]]:
        """Get the metrics for each run in the collection."""
        return [run.data.metrics for run in self._runs]

    @property
    def artifact_uri(self) -> list[str | None]:
        """Get the artifact URI for each run in the collection."""
        return [run.info.artifact_uri for run in self._runs]

    @property
    def artifact_dir(self) -> list[Path]:
        """Get the artifact directory for each run in the collection."""
        return [get_artifact_dir(run) for run in self._runs]

    @property
    def config(self) -> list[DictConfig]:
        """Get the configuration for each run in the collection."""
        return [load_config(run) for run in self._runs]


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
