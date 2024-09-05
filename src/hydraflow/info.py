from __future__ import annotations

from typing import TYPE_CHECKING

from hydraflow.mlflow import get_artifact_dir

if TYPE_CHECKING:
    from pathlib import Path

    from hydraflow.runs import RunCollection


class RunCollectionInfo:
    def __init__(self, runs: RunCollection):
        self._runs = runs

    @property
    def run_id(self) -> list[str]:
        return [run.info.run_id for run in self._runs]

    @property
    def params(self) -> list[dict[str, str]]:
        return [run.data.params for run in self._runs]

    @property
    def metrics(self) -> list[dict[str, float]]:
        return [run.data.metrics for run in self._runs]

    @property
    def artifact_uri(self) -> list[str | None]:
        return [run.info.artifact_uri for run in self._runs]

    @property
    def artifact_dir(self) -> list[Path]:
        return [get_artifact_dir(run) for run in self._runs]
