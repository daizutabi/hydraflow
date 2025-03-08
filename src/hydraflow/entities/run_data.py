"""Provide data about `RunCollection` instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hydraflow.core.config import iter_params
from hydraflow.core.io import load_config

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from pandas import DataFrame

    from .run_collection import RunCollection


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
        from pandas import DataFrame

        values = [dict(iter_params(load_config(r))) for r in self._runs]
        return DataFrame(values)


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
