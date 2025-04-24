from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from .run import Run
    from .run_collection import RunCollection

R = TypeVar("R", bound="Run[Any, Any]")
S = TypeVar("S", bound="RunCollection[Any]")


class GroupBy(Generic[R, S]):
    groups: dict[Any, S]

    def __init__(self, groups: dict[Any, S]) -> None:
        self.groups = groups

    def __getitem__(self, key: Any) -> S:
        return self.groups[key]

    # def group_by(self, *keys: str) -> DataFrame:
    #     """Group runs by one or more keys.

    #     This method returns a Polars DataFrame with group keys and aggregated values.

    #     Args:
    #         *keys (str): The keys to group by.

    #     Returns:
    #         DataFrame: A Polars DataFrame with group keys and aggregated values.

    #     """
    #     gp = self.group_by_dict(*keys)

    #     if len(keys) == 1:
    #         return DataFrame({keys[0]: list(gp)})

    #     return DataFrame(dict(zip(keys, k, strict=True)) for k in gp)
