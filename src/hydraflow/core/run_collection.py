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

    # Group runs by model type and compute aggregates
    grouped = runs.group_by("model.type",
                           avg_acc=lambda rc: sum(r.get("metrics.accuracy")
                                                 for r in rc) / len(rc))

    # Convert runs to a DataFrame for analysis
    df = runs.to_frame("run_id", "model.type", "metrics.accuracy")
    ```

Note:
    This module requires Polars and NumPy for DataFrame operations and
    numerical computations.

"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, overload

import numpy as np
import polars as pl
from omegaconf import OmegaConf
from polars import DataFrame, Series

from .run import Run

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Any, Self

    from numpy.typing import NDArray


class RunCollection[R: Run[Any, Any]](Sequence[R]):
    """A collection of Run instances that implements the Sequence protocol.

    RunCollection provides methods for filtering, sorting, grouping, and analyzing
    runs, as well as converting run data to various formats such as DataFrames.

    Args:
        runs (Iterable[Run]): An iterable of Run instances to include in
            the collection.

    """

    runs: list[R]
    """A list containing the Run instances in this collection."""

    def __init__(self, runs: Iterable[R]) -> None:
        self.runs = list(runs)

    def __repr__(self) -> str:
        """Return a string representation of the RunCollection."""
        class_name = self.__class__.__name__
        if not self:
            return f"{class_name}(empty)"

        type_name = repr(self[0])
        if "(" in type_name:
            type_name = type_name.split("(", 1)[0]
        return f"{class_name}({type_name}, n={len(self)})"

    def __len__(self) -> int:
        """Return the number of Run instances in the collection.

        Returns:
            int: The number of runs.

        """
        return len(self.runs)

    def __bool__(self) -> bool:
        """Return whether the collection contains any Run instances.

        Returns:
            bool: True if the collection is not empty, False otherwise.

        """
        return bool(self.runs)

    @overload
    def __getitem__(self, index: int) -> R: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    @overload
    def __getitem__(self, index: Iterable[int]) -> Self: ...

    def __getitem__(self, index: int | slice | Iterable[int]) -> R | Self:
        """Get a Run or a new RunCollection based on the provided index.

        Args:
            index: Can be one of:
                - An integer to get a single Run
                - A slice to get a subrange of Runs
                - An iterable of integers to get specific Runs

        Returns:
            R | Self: A single Run if index is an integer, or a new
            RunCollection if index is a slice or iterable of integers.

        """
        if isinstance(index, int):
            return self.runs[index]
        if isinstance(index, slice):
            return self.__class__(self.runs[index])
        return self.__class__([self.runs[i] for i in index])

    def __iter__(self) -> Iterator[R]:
        """Return an iterator over the Runs in the collection.

        Returns:
            Iterator[R]: An iterator yielding Run instances.

        """
        return iter(self.runs)

    def preload(
        self,
        *,
        n_jobs: int = 0,
        cfg: bool = True,
        impl: bool = True,
    ) -> Self:
        """Pre-load configuration and implementation objects for all runs in parallel.

        This method eagerly evaluates the cfg and impl properties of all runs
        in the collection, potentially in parallel using joblib. This can
        significantly improve performance for subsequent operations that
        access these properties, as they will be already loaded in memory.

        Args:
            cfg (bool): Whether to preload the configuration objects
            impl (bool): Whether to preload the implementation objects
            n_jobs (int): Number of parallel jobs to run
                (-1 means using all processors)

        Returns:
            Self: The same RunCollection instance with preloaded
            configuration and implementation objects.

        """

        def load(run: R) -> None:
            _ = cfg and run.cfg
            _ = impl and run.impl

        if n_jobs == 0:
            for run in self:
                load(run)
            return self

        from joblib import Parallel, delayed

        parallel = Parallel(backend="threading", n_jobs=n_jobs)
        parallel(delayed(load)(run) for run in self)
        return self

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

    def filter(
        self,
        *predicates: Callable[[R], bool] | tuple[str, Any],
        **kwargs: Any,
    ) -> Self:
        """Filter runs based on predicates or key-value conditions.

        This method allows filtering runs using various criteria:
        - Callable predicates that take a Run and return a boolean
        - Key-value tuples where the key is a string and the value
          is compared using the Run.predicate method
        - Keyword arguments, where the key is a string and the value
          is compared using the Run.predicate method

        Args:
            *predicates: Callable predicates or (key, value) tuples
                for filtering.
            **kwargs: Additional key-value pairs for filtering.

        Returns:
            Self: A new RunCollection containing only the runs that
            match all criteria.

        """
        runs = self.runs

        for predicate in predicates:
            if callable(predicate):
                runs = [r for r in runs if predicate(r)]
            else:
                runs = [r for r in runs if r.predicate(*predicate)]

        for key, value in kwargs.items():
            runs = [r for r in runs if r.predicate(key, value)]

        return self.__class__(runs)

    def try_get(
        self,
        *predicates: Callable[[R], bool] | tuple[str, Any],
        **kwargs: Any,
    ) -> R | None:
        """Try to get a single run matching the specified criteria.

        This method applies filters and returns a single matching
        run if exactly one is found, None if no runs are found,
        or raises ValueError if multiple runs match.

        Args:
            *predicates: Callable predicates or (key, value) tuples
                for filtering.
            **kwargs: Additional key-value pairs for filtering.

        Returns:
            R | None: A single Run that matches the criteria, or None if
            no matches are found.

        Raises:
            ValueError: If multiple runs match the criteria.

        """
        runs = self.filter(*predicates, **kwargs)

        n = len(runs)
        if n == 0:
            return None

        if n == 1:
            return runs[0]

        msg = f"Multiple Run ({n}) found matching the criteria, "
        msg += "expected exactly one"
        raise ValueError(msg)

    def get(
        self,
        *predicates: Callable[[R], bool] | tuple[str, Any],
        **kwargs: Any,
    ) -> R:
        """Get a single run matching the specified criteria.

        This method applies filters and returns a single matching run,
        or raises ValueError if no runs or multiple runs match.

        Args:
            *predicates: Callable predicates or (key, value) tuples
                for filtering.
            **kwargs: Additional key-value pairs for filtering.

        Returns:
            R: A single Run that matches the criteria.

        Raises:
            ValueError: If no runs match or if multiple runs match
            the criteria.

        """
        if run := self.try_get(*predicates, **kwargs):
            return run

        raise _value_error()

    def first(
        self,
        *predicates: Callable[[R], bool] | tuple[str, Any],
        **kwargs: Any,
    ) -> R:
        """Get the first run matching the specified criteria.

        This method applies filters and returns the first matching run,
        or raises ValueError if no runs match.

        Args:
            *predicates: Callable predicates or (key, value) tuples
                for filtering.
            **kwargs: Additional key-value pairs for filtering.

        Returns:
            R: The first Run that matches the criteria.

        Raises:
            ValueError: If no runs match the criteria.

        """
        if runs := self.filter(*predicates, **kwargs):
            return runs[0]

        raise _value_error()

    def last(
        self,
        *predicates: Callable[[R], bool] | tuple[str, Any],
        **kwargs: Any,
    ) -> R:
        """Get the last run matching the specified criteria.

        This method applies filters and returns the last matching run,
        or raises ValueError if no runs match.

        Args:
            *predicates: Callable predicates or (key, value) tuples
                for filtering.
            **kwargs: Additional key-value pairs for filtering.

        Returns:
            R: The last Run that matches the criteria.

        Raises:
            ValueError: If no runs match the criteria.

        """
        if runs := self.filter(*predicates, **kwargs):
            return runs[-1]

        raise _value_error()

    def to_list(
        self,
        key: str,
        default: Any | Callable[[R], Any] = MISSING,
    ) -> list[Any]:
        """Extract a list of values for a specific key from all runs.

        Args:
            key: The key to extract from each run.
            default: The default value to return if the key is not found.
                If a callable, it will be called with the Run instance
                and the value returned will be used as the default.

        Returns:
            list[Any]: A list containing the values for the
            specified key from each run.

        """
        return [run.get(key, default) for run in self]

    def to_numpy(
        self,
        key: str,
        default: Any | Callable[[R], Any] = MISSING,
    ) -> NDArray:
        """Extract values for a specific key from all runs as a NumPy array.

        Args:
            key: The key to extract from each run.
            default: The default value to return if the key is not found.
                If a callable, it will be called with the Run instance
                and the value returned will be used as the default.

        Returns:
            NDArray: A NumPy array containing the values for the
            specified key from each run.

        """
        return np.array(self.to_list(key, default))

    def to_series(
        self,
        key: str,
        default: Any | Callable[[R], Any] = MISSING,
        *,
        name: str | None = None,
    ) -> Series:
        """Extract values for a specific key from all runs as a Polars series.

        Args:
            key: The key to extract from each run.
            default: The default value to return if the key is not found.
                If a callable, it will be called with the Run instance
                and the value returned will be used as the default.
            name: The name of the series. If not provided, the key will be used.

        Returns:
            Series: A Polars series containing the values for the
            specified key from each run.

        """
        return Series(name or key, self.to_list(key, default))

    def unique(
        self,
        key: str,
        default: Any | Callable[[R], Any] = MISSING,
    ) -> NDArray:
        """Get the unique values for a specific key across all runs.

        Args:
            key: The key to extract unique values for.
            default: The default value to return if the key is not found.
                If a callable, it will be called with the Run instance
                and the value returned will be used as the default.

        Returns:
            NDArray: A NumPy array containing the unique values for the
            specified key.

        """
        return np.unique(self.to_numpy(key, default), axis=0)

    def n_unique(
        self,
        key: str,
        default: Any | Callable[[R], Any] = MISSING,
    ) -> int:
        """Count the number of unique values for a specific key across all runs.

        Args:
            key: The key to count unique values for.
            default: The default value to return if the key is not found.
                If a callable, it will be called with the Run instance
                and the value returned will be used as the default.

        Returns:
            int: The number of unique values for the specified key.

        """
        return len(self.unique(key, default))

    def sort(self, *keys: str, reverse: bool = False) -> Self:
        """Sort runs based on one or more keys.

        Args:
            *keys: The keys to sort by, in order of priority.
            reverse: Whether to sort in descending order (default is
                ascending).

        Returns:
            Self: A new RunCollection with the runs sorted according to
            the specified keys.

        """
        if not keys:
            return self

        arrays = [self.to_numpy(key) for key in keys]
        index = np.lexsort(arrays[::-1])

        if reverse:
            index = index[::-1]

        return self[index]

    def to_frame(
        self,
        *keys: str,
        defaults: dict[str, Any | Callable[[R], Any]] | None = None,
        **kwargs: Callable[[R], Any],
    ) -> DataFrame:
        """Convert the collection to a Polars DataFrame.

        Args:
            *keys (str): The keys to include as columns in the DataFrame.
                If not provided, all keys from each run's to_dict() method
                will be used.
            defaults (dict[str, Any | Callable[[R], Any]] | None): Default
                values for the keys. If a callable, it will be called with
                the Run instance and the value returned will be used as the
                default.
            **kwargs (Callable[[R], Any]): Additional columns to compute
                using callables that take a Run and return a value.

        Returns:
            DataFrame: A Polars DataFrame containing the specified data
            from the runs.

        """
        if defaults is None:
            defaults = {}

        if keys:
            df = DataFrame(
                {key: self.to_list(key, defaults.get(key, MISSING)) for key in keys},
            )
        else:
            df = DataFrame(r.to_dict() for r in self)

        if not kwargs:
            return df

        columns = [Series(k, [v(r) for r in self]) for k, v in kwargs.items()]
        return df.with_columns(*columns)

    def _group_by(self, *keys: str) -> dict[Any, Self]:
        result: dict[Any, Self] = {}

        for run in self:
            keys_ = [to_hashable(run.get(key)) for key in keys]
            key = keys_[0] if len(keys) == 1 else tuple(keys_)

            if key not in result:
                result[key] = self.__class__([])
            result[key].runs.append(run)

        return result

    @overload
    def group_by(self, *keys: str) -> dict[Any, Self]: ...

    @overload
    def group_by(
        self,
        *keys: str,
        **kwargs: Callable[[Self | Sequence[R]], Any],
    ) -> DataFrame: ...

    def group_by(
        self,
        *keys: str,
        **kwargs: Callable[[Self | Sequence[R]], Any],
    ) -> dict[Any, Self] | DataFrame:
        """Group runs by one or more keys.

        This method can return either:
        - A dictionary mapping group keys to RunCollections
          (no kwargs provided)
        - A Polars DataFrame with group keys and aggregated
          values (kwargs provided)

        Args:
            *keys (str): The keys to group by.
            **kwargs (Callable[[Self | Sequence[R]], Any]): Aggregation
                functions to apply to each group. Each function should
                accept a RunCollection or Sequence[Run] and return a value.

        Returns:
            dict[Any, Self] | DataFrame: Either a dictionary mapping
            group keys to RunCollections, or a Polars DataFrame with
            group keys and aggregated values.

        """
        gp = self._group_by(*keys)
        if not kwargs:
            return gp

        if len(keys) == 1:
            df = DataFrame({keys[0]: list(gp)})
        else:
            df = DataFrame(dict(zip(keys, k, strict=True)) for k in gp)
        columns = [pl.Series(k, [v(r) for r in gp.values()]) for k, v in kwargs.items()]
        return df.with_columns(*columns)


def to_hashable(value: Any) -> Hashable:
    """Convert a value to a hashable instance.

    This function handles various types of values and converts them to
    hashable equivalents for use in dictionaries and sets.

    Args:
        value: The value to convert to a hashable instance.

    Returns:
        A hashable version of the input value.

    """
    if OmegaConf.is_list(value):  # Is ListConfig hashable?
        return tuple(value)
    if isinstance(value, Hashable):
        return value
    if isinstance(value, np.ndarray):
        return tuple(value.tolist())
    try:
        return tuple(value)
    except TypeError:
        return str(value)


def _value_error() -> ValueError:
    msg = "No Run found matching the specified criteria"
    return ValueError(msg)
