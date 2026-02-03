"""GroupBy module for organizing and aggregating collections of items.

This module provides the GroupBy class, which represents the result of a
group_by operation on a Collection. It organizes items into groups based on
specified keys and enables aggregation operations across those groups.

The GroupBy class implements a dictionary-like interface, allowing access to
individual groups through key lookup, iteration, and standard dictionary
methods like keys(), values(), and items().

Example:
    ```python
    # Group runs by model type
    grouped = runs.group_by("model.type")

    # Access a specific group
    transformer_runs = grouped["transformer"]

    # Iterate through groups
    for model_type, group in grouped.items():
        print(f"Model: {model_type}, Runs: {len(group)}")

    # Perform aggregations
    stats = grouped.agg(
        "accuracy",
        "loss",
        avg_time=lambda g: sum(r.get("runtime") for r in g) / len(g)
    )
    ```

The GroupBy class supports aggregation through the agg() method, which can
compute both predefined metrics from the grouped items and custom aggregations
specified as callables.

"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from polars import DataFrame, Series

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        ItemsView,
        Iterator,
        KeysView,
        Sequence,
        ValuesView,
    )

    from .collection import Collection


class GroupBy[C: Collection[Any], I]:
    """Represents the result of a group_by operation on a Collection.

    The GroupBy class organizes items from a Collection into groups based on
    specified keys. It provides a dictionary-like interface for accessing the
    groups and methods for aggregating data across the groups.

    Attributes:
        by: The keys used for grouping.
        groups: A dictionary mapping group keys to Collection instances.

    """

    by: tuple[str, ...]
    groups: dict[Any, C]

    def __init__(self, by: tuple[str, ...], groups: dict[Any, C]) -> None:
        """Initialize a GroupBy instance.

        Args:
            by: The keys used for grouping.
            groups: A dictionary mapping group keys to Collection instances.

        """
        self.by = by
        self.groups = groups

    def __getitem__(self, key: Any) -> C:
        """Get a group by its key.

        Args:
            key: The group key to look up.

        Returns:
            The Collection corresponding to the key.

        """
        return self.groups[key]

    def __iter__(self) -> Iterator[Any]:
        """Iterate over group keys.

        Returns:
            An iterator over the group keys.

        """
        return iter(self.groups)

    def __len__(self) -> int:
        """Get the number of groups.

        Returns:
            The number of groups.

        """
        return len(self.groups)

    def __contains__(self, key: Any) -> bool:
        """Check if a key is in the groups.

        Args:
            key: The key to check for.

        Returns:
            True if the key is in the groups, False otherwise.

        """
        return key in self.groups

    def keys(self) -> KeysView[Any]:
        """Get the keys of the groups.

        Returns:
            A view of the group keys.

        """
        return self.groups.keys()

    def values(self) -> ValuesView[C]:
        """Get the values (Collections) of the groups.

        Returns:
            A view of the group values.

        """
        return self.groups.values()

    def items(self) -> ItemsView[Any, C]:
        """Get the (key, value) pairs of the groups.

        Returns:
            A view of the (key, value) pairs.

        """
        return self.groups.items()

    def agg(
        self,
        *aggs: str,
        **named_aggs: Callable[[C | Sequence[I]], Any],
    ) -> DataFrame:
        """Aggregate data across groups.

        This method computes aggregations for each group and returns the results
        as a DataFrame. There are two ways to specify aggregations:

        1. String keys: These are interpreted as attributes to extract from each
           item in the group.
        2. Callables: Functions that take a Collection or Sequence of items and
           return an aggregated value.

        Args:
            *aggs: String keys to aggregate.
            **named_aggs: Named aggregation functions.

        Returns:
            A DataFrame with group keys and aggregated values.

        Example:
            ```python
            # Aggregate by accuracy and loss, and compute average runtime
            stats = grouped.agg(
                "accuracy",
                "loss",
                avg_runtime=lambda g: sum(r.get("runtime") for r in g) / len(g)
            )
            ```

        """
        gp = self.groups

        if len(self.by) == 1:
            df = DataFrame({self.by[0]: list(gp)})
        else:
            df = DataFrame(dict(zip(self.by, k, strict=True)) for k in gp)

        columns: list[Series] = []

        for agg in aggs:
            values = [[c._get(i, agg, MISSING) for i in c] for c in gp.values()]  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            columns.append(Series(agg, values))

        for k, v in named_aggs.items():
            columns.append(Series(k, [v(r) for r in gp.values()]))

        return df.with_columns(columns)
