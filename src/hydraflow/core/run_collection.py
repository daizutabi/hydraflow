"""RunCollection module for HydraFlow.

This module provides the RunCollection class, which represents a collection
of MLflow Runs in HydraFlow. RunCollection offers functionality for filtering,
sorting, grouping, and analyzing runs, as well as converting run data to
various formats such as DataFrames.

The RunCollection class implements the Sequence protocol, allowing it to be
used like a standard Python list while providing specialized methods for
working with Run instances.

Example:
    ```python
    # Create a collection from a list of runs
    runs = RunCollection([run1, run2, run3])

    # Filter runs based on criteria
    filtered = runs.filter(("metrics.accuracy", lambda acc: acc > 0.9))

    # Sort runs by specific keys
    sorted_runs = runs.sort("metrics.accuracy", reverse=True)

    # Group runs by model type
    grouped = runs.group_by("model.type")

    # Compute aggregates on grouped data
    metrics_df = grouped.agg(
        avg_acc=lambda rc: sum(r.get("metrics.accuracy") for r in rc) / len(rc)
    )

    # Convert runs to a DataFrame for analysis
    df = runs.to_frame("run_id", "model.type", "metrics.accuracy")
    ```

Note:
    This module requires Polars and NumPy for DataFrame operations and
    numerical computations.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import polars as pl

from .collection import Collection
from .run import Run

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

    from polars import DataFrame

# pyright: reportUnknownVariableType=false


class RunCollection[R: Run[Any, Any]](Collection[R]):
    """A collection of Run instances that implements the Sequence protocol.

    RunCollection provides methods for filtering, sorting, grouping, and analyzing
    runs, as well as converting run data to various formats such as DataFrames.

    Args:
        runs (Iterable[Run]): An iterable of Run instances to include in
            the collection.

    """

    @overload
    def update(
        self,
        key: str,
        value: Any | Callable[[R], Any],
        *,
        force: bool = False,
    ) -> None: ...

    @overload
    def update(
        self,
        key: tuple[str, ...],
        value: Iterable[Any] | Callable[[R], Iterable[Any]],
        *,
        force: bool = False,
    ) -> None: ...

    def update(
        self,
        key: str | tuple[str, ...],
        value: Any | Callable[[R], Any],
        *,
        force: bool = False,
    ) -> None:
        """Update configuration values for all runs in the collection.

        This method calls the update method on each run in the collection.

        Args:
            key: Either a string representing a single configuration path
                or a tuple of strings to set multiple configuration values.
            value: The value(s) to set or a callable that returns such values.
            force: Whether to force updates even if the keys already exist.

        """
        for run in self:
            run.update(key, value, force=force)

    def concat(
        self,
        function: Callable[[R], DataFrame],
        *keys: str | tuple[str, Any | Callable[[R], Any]],
    ) -> DataFrame:
        """Concatenate the results of a function applied to all runs in the collection.

        This method applies the provided function to each run in the collection
        and concatenates the resulting DataFrames along the specified keys.

        Args:
            function (Callable[[R], DataFrame]): A function that takes a Run
                instance and returns a DataFrame.
            keys (str | tuple[str, Any | Callable[[R], Any]]): The keys to
                add to the DataFrame.

        Returns:
            DataFrame: A DataFrame representation of the Run collection.

        """
        return pl.concat(run.to_frame(function, *keys) for run in self)

    def iterdir(self, relative_dir: str = "") -> Iterator[Path]:
        """Iterate over the artifact directories for all runs in the collection.

        This method yields all files and directories in the specified
        relative directory for each run in the collection.

        Args:
            relative_dir (str): The relative directory within the artifacts
                directory to iterate over.

        Yields:
            Path: Each path in the specified directory for each run
            in the collection.

        """
        for run in self:
            yield from run.iterdir(relative_dir)

    def glob(self, pattern: str, relative_dir: str = "") -> Iterator[Path]:
        """Glob the artifact directories for all runs in the collection.

        This method yields all paths matching the specified pattern
        in the relative directory for each run in the collection.

        Args:
            pattern (str): The glob pattern to match files or directories.
            relative_dir (str): The relative directory within the artifacts
                directory to search in.

        Yields:
            Path: Each path matching the pattern for each run in the collection.

        """
        for run in self:
            yield from run.glob(pattern, relative_dir)
