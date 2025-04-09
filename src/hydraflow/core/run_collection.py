from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, overload

import numpy as np
from omegaconf import OmegaConf

from .run import Run

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Any, Self

    from numpy.typing import NDArray


class RunCollection[R: Run]:
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

    def filter(
        self,
        predicate: Callable[[R], bool] | dict[str, Any] | None = None,
        **kwargs,
    ) -> Self:
        """Filter the Run instances based on the provided criteria.

        This method creates a new RunCollection containing only the Run
        instances that match the specified criteria. Filtering can be done using
        a custom predicate function, attribute-based filters, or a combination
        of both.

        Args:
            predicate: Optional function that takes a Run and returns True/False.
                If provided, this function is used to filter the runs.
            **kwargs: Attribute-based filters, where keys are attribute names
                and values are the filtering criteria.
                The following value types are supported:

                    - Simple value: Exact match (attr == value)
                    - List/Set: Membership check (attr in value)
                    - Tuple of 2 values: Range check (value[0] <= attr <= value[1])
                    - Callable: Custom evaluation function (value(attr) == True)

        Returns:
            A new RunCollection containing only the Run instances that match
            the specified criteria.

        """
        runs = [r for r in self if predicate(r)] if callable(predicate) else self.runs

        if isinstance(predicate, dict):
            kwargs |= predicate

        for name, value in kwargs.items():
            runs = [r for r in runs if _predicate(r, name, value)]

        return self.__class__(runs)

    def try_get(
        self,
        predicate: Callable[[R], bool] | dict[str, Any] | None = None,
        **kwargs,
    ) -> R | None:
        """Attempt to get a single Run instance that matches the criteria.

        This method returns a single matching Run if exactly one is found,
        returns None if no matches are found, and raises an error if multiple
        matches are found.

        Args:
            predicate: Optional function that takes a Run and returns True/False.
            **kwargs: Attribute-based filters (see filter method for details).

        Returns:
            The single matching Run instance, or None if no matches found.

        Raises:
            ValueError: If multiple matches are found.

        """
        runs = self.filter(predicate, **kwargs)

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
        predicate: Callable[[R], bool] | dict[str, Any] | None = None,
        **kwargs,
    ) -> R:
        """Get a single Run instance that matches the criteria.

        This method enforces that exactly one Run matches the given criteria.
        It raises an error if either no matches or multiple matches are found.

        Args:
            predicate: Optional function that takes a Run and returns True/False.
            **kwargs: Attribute-based filters (see filter method for details).

        Returns:
            The single matching Run instance.

        Raises:
            ValueError: If no matches or multiple matches are found.

        """
        if run := self.try_get(predicate, **kwargs):
            return run

        msg = "No Run found matching the specified criteria"
        raise ValueError(msg)

    def first(
        self,
        predicate: Callable[[R], bool] | dict[str, Any] | None = None,
        **kwargs,
    ) -> R:
        """Get the first Run instance that matches the criteria.

        Return the first Run that matches the given criteria.

        Args:
            predicate: Optional function that takes a Run and returns True/False.
            **kwargs: Attribute-based filters (see filter method for details).

        Returns:
            The first matching Run instance.

        Raises:
            ValueError: If no matches are found.

        """
        if runs := self.filter(predicate, **kwargs):
            return runs[0]

        msg = "No Run found matching the specified criteria"
        raise ValueError(msg)

    def last(
        self,
        predicate: Callable[[R], bool] | dict[str, Any] | None = None,
        **kwargs,
    ) -> R:
        """Get the last Run instance that matches the criteria.

        Return the last Run that matches the given criteria.

        Args:
            predicate: Optional function that takes a Run and returns True/False.
            **kwargs: Attribute-based filters (see filter method for details).

        Returns:
            The last matching Run instance.

        Raises:
            ValueError: If no matches are found.

        """
        if runs := self.filter(predicate, **kwargs):
            return runs[-1]

        msg = "No Run found matching the specified criteria"
        raise ValueError(msg)

    def to_list(self, key: str) -> list[Any]:
        """Return a list of the specified attribute values from all runs.

        This method collects the specified attribute from each
        Run in the collection and returns them as a list.
        The attribute can be a nested attribute using dot notation

        Args:
            key: The key of the attribute to collect. Can use dot notation
                for nested attributes.

        Returns:
            list: A list containing the attribute values from all runs.

        Raises:
            AttributeError: If the specified attribute is not found in any run.

        """
        return [_getattr(run, key) for run in self]

    def to_numpy(self, key: str) -> NDArray:
        """Return a numpy array of the specified attribute values from all runs.

        This method collects the specified attribute from each
        Run in the collection and returns them as a numpy array.
        The attribute can be a nested attribute using dot notation

        Args:
            key (str): The key of the attribute to collect.
                Can use dot notation for nested attributes
                (e.g., 'cfg.size').

        Returns:
            NDArray: A numpy array containing the attribute values from all runs.

        Raises:
            AttributeError: If the specified attribute is not found in any run.

        """
        return np.array(self.to_list(key))

    def unique(self, key: str) -> NDArray:
        """Return the unique values of the specified attribute across all runs.

        This method collects the specified attribute from each Run
        in the collection, then returns the unique values as a sorted
        numpy array.

        Args:
            key (str): The key of the attribute to collect.
                Can use dot notation for nested attributes
                (e.g., 'cfg.size').

        Returns:
            NDArray: A sorted numpy array containing the unique attribute values.

        Raises:
            AttributeError: If the specified attribute is not found in any run.

        """
        return np.unique(self.to_numpy(key), axis=0)

    def n_unique(self, key: str) -> int:
        """Return the number of unique values for the specified attribute.

        Args:
            key (str): The key of the attribute to analyze.
                Can use dot notation for nested attributes.

        Returns:
            int: The number of unique values for the specified attribute.

        """
        return len(self.unique(key))

    def sorted(self, *keys: str, reverse: bool = False) -> Self:
        """Return a new collection with Run instances sorted by attributes.

        This method returns a new RunCollection with Run
        instances sorted according to the specified attributes.
        When multiple attributes are provided, sorting is performed
        hierarchically (first by the first attribute, then by
        the second for ties, and so on). If no attributes are provided,
        the original collection is returned unchanged.

        Args:
        *keys: The attribute keys to sort by. Can use dot
            notation for nested attributes (e.g., 'cfg.size').
            Multiple attributes can be specified for hierarchical sorting.
        reverse: If True, sort in descending order. Default is
            False (ascending).

        Returns:
            Self: A new RunCollection with the same Run instances,
            but in sorted order. If no names are provided, returns self.

        """
        if not keys:
            return self

        arrays = [self.to_numpy(key) for key in keys]
        index = np.lexsort(arrays[::-1])

        if reverse:
            index = index[::-1]

        return self[index]

    def group_by(self, *keys: str) -> dict[Any, Self]:
        """Group Run instances by the values of the specified attributes.

        Args:
            *keys (str): The keys of the attributes to group by.
                Can use dot notation.

        Returns:
            dict[Any, Self]: A dictionary mapping attribute value tuples to
            RunCollection instances.

        """
        result: dict[Any, Self] = {}

        for run in self:
            keys_ = [to_hashable(_getattr(run, key)) for key in keys]
            key = keys_[0] if len(keys) == 1 else tuple(keys_)

            if key not in result:
                result[key] = self.__class__([])
            result[key].runs.append(run)

        return result

    @overload
    def set_default(
        self,
        key: str,
        value: Any | Callable[[R], Any],
    ) -> None: ...

    @overload
    def set_default(
        self,
        key: tuple[str, ...],
        value: Iterable[Any] | Callable[[R], Iterable[Any]],
    ) -> None: ...

    def set_default(
        self,
        key: str | tuple[str, ...],
        value: Any | Callable[[R], Any],
    ) -> None:
        """Set default value(s) in the configuration of all RunConfig instances.

        This method calls set_default on each RunConfig instance in the collection,
        applying the same key and value to all of them.

        Args:
            key: Either a string representing a single configuration path
                (can use dot notation like "section.subsection.param"),
                or a tuple of strings to set multiple related configuration
                values at once.
            value: The value to set. This can be:
                - For string keys: Any value, or a callable that takes a
                  RunConfig instance and returns a value
                - For tuple keys: An iterable with the same length as the
                  key tuple, or a callable that returns such an iterable

        Raises:
            TypeError: If a tuple key is provided but the value is not an
                iterable, or if the callable doesn't return an iterable.

        """
        for run in self:
            run.set_default(key, value)


def _getattr(obj: Any, key: str) -> Any:
    if "." in key:
        first, key = key.split(".", 1)
        obj = _getattr(obj, first)

    if hasattr(obj, key):
        attr = getattr(obj, key)

        if callable(attr):
            return attr()

        return attr

    if isinstance(obj, Run) and hasattr(obj.cfg, key):
        return _getattr(obj.cfg, key)

    msg = f"Attribute not found: {key}"
    raise AttributeError(msg)


def _predicate(obj: Any, key: str, value: Any) -> bool:
    attr = _getattr(obj, key)

    if callable(value):
        return bool(value(attr))

    if isinstance(value, list | set) and not isinstance(attr, list | set):
        return attr in value

    if isinstance(value, tuple) and len(value) == 2 and not isinstance(attr, tuple):
        return value[0] <= attr <= value[1]

    return attr == value


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
