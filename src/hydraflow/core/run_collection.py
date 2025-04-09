from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, overload

import numpy as np
import polars as pl
from omegaconf import OmegaConf
from polars import DataFrame

from .run import Run

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Any, Self

    from numpy.typing import NDArray


class RunCollection[R: Run[Any, Any]](Sequence[R]):
    runs: list[R]

    def __init__(self, runs: Iterable[R]) -> None:
        self.runs = list(runs)

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        if not self:
            return f"{class_name}(empty)"

        type_name = self[0].__class__.__name__
        return f"{class_name}({type_name}, n={len(self)})"

    def __len__(self) -> int:
        return len(self.runs)

    def __bool__(self) -> bool:
        return bool(self.runs)

    @overload
    def __getitem__(self, index: int) -> R: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    @overload
    def __getitem__(self, index: Iterable[int]) -> Self: ...

    def __getitem__(self, index: int | slice | Iterable[int]) -> R | Self:
        if isinstance(index, int):
            return self.runs[index]
        if isinstance(index, slice):
            return self.__class__(self.runs[index])
        return self.__class__([self.runs[i] for i in index])

    def __iter__(self) -> Iterator[R]:
        return iter(self.runs)

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
        for run in self:
            run.update(key, value, force=force)

    def filter(
        self,
        *predicates: Callable[[R], bool] | tuple[str, Any],
        **kwargs: Any,
    ) -> Self:
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
        if run := self.try_get(*predicates, **kwargs):
            return run

        msg = "No Run found matching the specified criteria"
        raise ValueError(msg)

    def first(
        self,
        *predicates: Callable[[R], bool] | tuple[str, Any],
        **kwargs: Any,
    ) -> R:
        if runs := self.filter(*predicates, **kwargs):
            return runs[0]

        msg = "No Run found matching the specified criteria"
        raise ValueError(msg)

    def last(
        self,
        *predicates: Callable[[R], bool] | tuple[str, Any],
        **kwargs: Any,
    ) -> R:
        if runs := self.filter(*predicates, **kwargs):
            return runs[-1]

        msg = "No Run found matching the specified criteria"
        raise ValueError(msg)

    def to_list(self, key: str) -> list[Any]:
        return [run.get(key) for run in self]

    def to_numpy(self, key: str) -> NDArray:
        return np.array(self.to_list(key))

    def unique(self, key: str) -> NDArray:
        return np.unique(self.to_numpy(key), axis=0)

    def n_unique(self, key: str) -> int:
        return len(self.unique(key))

    def sort(self, *keys: str, reverse: bool = False) -> Self:
        if not keys:
            return self

        arrays = [self.to_numpy(key) for key in keys]
        index = np.lexsort(arrays[::-1])

        if reverse:
            index = index[::-1]

        return self[index]

    def to_frame(self, *keys: str, **kwargs: Callable[[R], Any]) -> DataFrame:
        if keys:
            df = DataFrame({key: self.to_list(key) for key in keys})
        else:
            df = DataFrame(r.to_dict() for r in self)

        if not kwargs:
            return df

        columns = [pl.Series(k, [v(r) for r in self]) for k, v in kwargs.items()]
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
        gp = self._group_by(*keys)
        if not kwargs:
            return gp

        df = DataFrame(dict(zip(keys, k, strict=True)) for k in gp)
        columns = [pl.Series(k, [v(r) for r in gp.values()]) for k, v in kwargs.items()]
        return df.with_columns(*columns)


def to_hashable(value: Any) -> Hashable:
    """Convert value to hashable."""
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
