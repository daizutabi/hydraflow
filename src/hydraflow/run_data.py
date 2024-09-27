"""Provide data about `RunCollection` instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hydraflow.utils import load_config

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from hydraflow.run_collection import RunCollection


class RunCollectionData:
    """Provide data about a `RunCollection` instance."""

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
