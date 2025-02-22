"""Provide information about `RunCollection` instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hydraflow.core.io import get_artifact_dir

if TYPE_CHECKING:
    from pathlib import Path

    from .run_collection import RunCollection


class RunCollectionInfo:
    """Provide information about a `RunCollection` instance."""

    _runs: RunCollection

    def __init__(self, runs: RunCollection) -> None:
        self._runs = runs

    @property
    def run_id(self) -> list[str]:
        """Get the run ID for each run in the collection."""
        return [run.info.run_id for run in self._runs]

    @property
    def artifact_uri(self) -> list[str | None]:
        """Get the artifact URI for each run in the collection."""
        return [run.info.artifact_uri for run in self._runs]

    @property
    def artifact_dir(self) -> list[Path]:
        """Get the artifact directory for each run in the collection."""
        return [get_artifact_dir(run) for run in self._runs]
