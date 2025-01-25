"""Provide data about `RunCollection` instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame

from hydraflow.config import collect_params

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from hydraflow.run_collection import RunCollection


class RunCollectionData:
    """Provide data about a `RunCollection` instance."""

    def __init__(self, runs: RunCollection) -> None:
        self._runs = runs

    @property
    def params(self) -> dict[str, list[str]]:
        """Get the parameters for each run in the collection."""
        return _to_dict(run.data.params for run in self._runs)

    @property
    def metrics(self) -> dict[str, list[float]]:
        """Get the metrics for each run in the collection."""
        return _to_dict(run.data.metrics for run in self._runs)

    @property
    def config(self) -> DataFrame:
        """Get the runs' configurations as a DataFrame.

        Returns:
            A DataFrame containing the runs' configurations.

        """
        return DataFrame(self._runs.map_config(collect_params))


def _to_dict(it: Iterable[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert an iterable of dictionaries to a dictionary of lists."""
    data = list(it)
    if not data:
        return {}

    keys = []
    for d in data:
        for key in d:
            if key not in keys:
                keys.append(key)

    return {key: [x.get(key) for x in data] for key in keys}
