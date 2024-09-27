"""Provide information about MLflow runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

from hydraflow.run_info import get_artifact_dir

if TYPE_CHECKING:
    from mlflow.entities import Run

    from hydraflow.run_collection import RunCollection


class RunCollectionData:
    """Provide information about MLflow runs."""

    def __init__(self, runs: RunCollection) -> None:
        self._runs = runs

    @property
    def params(self) -> list[dict[str, str]]:
        """Get the parameters for each run in the collection."""
        return [run.data.params for run in self._runs]

    @property
    def metrics(self) -> list[dict[str, float]]:
        """Get the metrics for each run in the collection."""
        return [run.data.metrics for run in self._runs]

    @property
    def config(self) -> list[DictConfig]:
        """Get the configuration for each run in the collection."""
        return [load_config(run) for run in self._runs]


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
