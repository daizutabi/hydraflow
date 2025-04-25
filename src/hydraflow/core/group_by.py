from __future__ import annotations

from typing import TYPE_CHECKING, Any

from polars import DataFrame, Series

if TYPE_CHECKING:
    from collections.abc import Callable, ItemsView, Iterator, KeysView, ValuesView

    from .collection import Collection


class GroupBy[C: Collection[Any]]:
    by: tuple[str, ...]
    groups: dict[Any, C]

    def __init__(self, by: tuple[str, ...], groups: dict[Any, C]) -> None:
        self.by = by
        self.groups = groups

    def __getitem__(self, key: Any) -> C:
        return self.groups[key]

    def __iter__(self) -> Iterator[C]:
        return iter(self.groups.values())

    def __len__(self) -> int:
        return len(self.groups)

    def __contains__(self, key: Any) -> bool:
        return key in self.groups

    def keys(self) -> KeysView[Any]:
        return self.groups.keys()

    def values(self) -> ValuesView[C]:
        return self.groups.values()

    def items(self) -> ItemsView[Any, C]:
        return self.groups.items()

    def agg(self, *aggs: str, run: Callable[[R], Any] | None = None) -> DataFrame:
        gp = self.groups

        if len(self.by) == 1:
            df = DataFrame({self.by[0]: list(gp)})
        else:
            df = DataFrame(dict(zip(self.by, k, strict=True)) for k in gp)

        columns = []

        for agg in aggs:
            if agg == "run":
                columns.append(Series("run", list(gp.values())))
            else:
                values = [[r.get(agg) for r in rc] for rc in gp.values()]
                columns.append(Series(agg, values))

        df = df.with_columns(columns)

        return df
