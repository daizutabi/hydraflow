"""Provide a collection of items that implements the Sequence protocol."""

from __future__ import annotations

import random
import re
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Concatenate, overload

import numpy as np
from joblib.parallel import Parallel, delayed
from omegaconf import ListConfig, OmegaConf
from polars import DataFrame, Series

from .group_by import GroupBy

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Any, Self

    from numpy.typing import NDArray


class Collection[I](Sequence[I]):
    """A collection of items that implements the Sequence protocol."""

    _items: list[I]
    _get: Callable[[I, str, Any | Callable[[I], Any]], Any]

    def __init__(
        self,
        items: Iterable[I],
        get: Callable[[I, str, Any | Callable[[I], Any]], Any] | None = None,
    ) -> None:
        self._items = list(items)
        self._get = get or getattr

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        if not self:
            return f"{class_name}(empty)"

        type_name = repr(self[0])
        if "(" in type_name:
            type_name = type_name.split("(", 1)[0]
        return f"{class_name}({type_name}, n={len(self)})"

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    @overload
    def __getitem__(self, index: int) -> I: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    @overload
    def __getitem__(self, index: Iterable[int]) -> Self: ...

    def __getitem__(self, index: int | slice | Iterable[int]) -> I | Self:
        if isinstance(index, int):
            return self._items[index]

        if isinstance(index, slice):
            return self.__class__(self._items[index], self._get)

        return self.__class__([self._items[i] for i in index], self._get)

    def __iter__(self) -> Iterator[I]:
        return iter(self._items)

    def filter(
        self,
        *criteria: Callable[[I], bool] | tuple[str, Any],
        **kwargs: Any,
    ) -> Self:
        """Filter items based on criteria.

        This method allows filtering items using various criteria:

        - Callable criteria that take an item and return a boolean
        - Key-value tuples where the key is a string and the value
          is compared using the `matches` function
        - Keyword arguments, where the key is a string and the value
          is compared using the `matches` function

        The `matches` function supports the following comparison types:

        - Callable: The predicate function is called with the value
        - List/Set: Checks if the value is in the list/set
        - Tuple of length 2: Checks if the value is in the range [min, max]
        - Other: Checks for direct equality

        Args:
            *criteria: Callable criteria or (key, value) tuples
                for filtering.
            **kwargs: Additional key-value pairs for filtering.

        Returns:
            Self: A new Collection containing only the items that
            match all criteria.

        Examples:
            ```python
            # Filter using a callable
            filtered = collection.filter(lambda x: x > 5)

            # Filter using a key-value tuple
            filtered = collection.filter(("age", 25))

            # Filter using keyword arguments
            filtered = collection.filter(age=25, name="John")

            # Filter using range
            filtered = collection.filter(("age", (20, 30)))

            # Filter using list membership
            filtered = collection.filter(("name", ["John", "Jane"]))
            ```

        """
        items = self._items

        for c in criteria:
            if callable(c):
                items = [i for i in items if c(i)]
            else:
                items = [i for i in items if matches(self._get(i, c[0], MISSING), c[1])]

        for key, value in kwargs.items():
            items = [i for i in items if matches(self._get(i, key, MISSING), value)]

        return self.__class__(items, self._get)

    def try_get(
        self,
        *criteria: Callable[[I], bool] | tuple[str, Any],
        **kwargs: Any,
    ) -> I | None:
        """Try to get a single item matching the specified criteria.

        This method applies filters and returns a single matching
        item if exactly one is found, None if no items are found,
        or raises ValueError if multiple items match.

        Args:
            *criteria: Callable criteria or (key, value) tuples
                for filtering.
            **kwargs: Additional key-value pairs for filtering.

        Returns:
            I | None: A single item that matches the criteria, or None if
            no matches are found.

        Raises:
            ValueError: If multiple items match the criteria.

        """
        items = self.filter(*criteria, **kwargs)

        n = len(items)
        if n == 0:
            return None

        if n == 1:
            return items[0]

        msg = f"Multiple items ({n}) found matching the criteria, "
        msg += "expected exactly one"
        raise ValueError(msg)

    def get(
        self,
        *criteria: Callable[[I], bool] | tuple[str, Any],
        **kwargs: Any,
    ) -> I:
        """Get a single item matching the specified criteria.

        This method applies filters and returns a single matching item,
        or raises ValueError if no items or multiple items match.

        Args:
            *criteria: Callable criteria or (key, value) tuples
                for filtering.
            **kwargs: Additional key-value pairs for filtering.

        Returns:
            I: A single item that matches the criteria.

        Raises:
            ValueError: If no items match or if multiple items match
            the criteria.

        """
        if item := self.try_get(*criteria, **kwargs):
            return item

        raise _value_error()

    def first(
        self,
        *criteria: Callable[[I], bool] | tuple[str, Any],
        **kwargs: Any,
    ) -> I:
        """Get the first item matching the specified criteria.

        This method applies filters and returns the first matching item,
        or raises ValueError if no items match.

        Args:
            *criteria: Callable criteria or (key, value) tuples
                for filtering.
            **kwargs: Additional key-value pairs for filtering.

        Returns:
            I: The first item that matches the criteria.

        Raises:
            ValueError: If no items match the criteria.

        """
        if items := self.filter(*criteria, **kwargs):
            return items[0]

        raise _value_error()

    def last(
        self,
        *criteria: Callable[[I], bool] | tuple[str, Any],
        **kwargs: Any,
    ) -> I:
        """Get the last item matching the specified criteria.

        This method applies filters and returns the last matching item,
        or raises ValueError if no items match.

        Args:
            *criteria: Callable criteria or (key, value) tuples
                for filtering.
            **kwargs: Additional key-value pairs for filtering.

        Returns:
            I: The last item that matches the criteria.

        Raises:
            ValueError: If no items match the criteria.

        """
        if items := self.filter(*criteria, **kwargs):
            return items[-1]

        raise _value_error()

    def to_list(
        self,
        key: str,
        default: Any | Callable[[I], Any] = MISSING,
    ) -> list[Any]:
        """Extract a list of values for a specific key from all items.

        Args:
            key: The key to extract from each item.
            default: The default value to return if the key is not found.
                If a callable, it will be called with the item
                and the value returned will be used as the default.

        Returns:
            list[Any]: A list containing the values for the
            specified key from each item.

        """
        return [self._get(i, key, default) for i in self]

    def to_numpy(
        self,
        key: str,
        default: Any | Callable[[I], Any] = MISSING,
    ) -> NDArray:
        """Extract values for a specific key from all items as a NumPy array.

        Args:
            key: The key to extract from each item.
            default: The default value to return if the key is not found.
                If a callable, it will be called with the item
                and the value returned will be used as the default.

        Returns:
            NDArray: A NumPy array containing the values for the
            specified key from each item.

        """
        return np.array(self.to_list(key, default))

    def to_series(
        self,
        key: str,
        default: Any = MISSING,
        *,
        name: str | None = None,
    ) -> Series:
        """Extract values for a specific key from all items as a Polars series.

        Args:
            key: The key to extract from each item.
            default: The default value to return if the key is not found.
                If a callable, it will be called with the item
                and the value returned will be used as the default.
            name: The name of the series. If not provided, the key will be used.

        Returns:
            Series: A Polars series containing the values for the
            specified key from each item.

        """
        return Series(name or key, self.to_list(key, default))

    def unique(
        self,
        key: str,
        default: Any | Callable[[I], Any] = MISSING,
    ) -> NDArray:
        """Get the unique values for a specific key across all items.

        Args:
            key: The key to extract unique values for.
            default: The default value to return if the key is not found.
                If a callable, it will be called with the item
                and the value returned will be used as the default.

        Returns:
            NDArray: A NumPy array containing the unique values for the
            specified key.

        """
        return np.unique(self.to_numpy(key, default), axis=0)

    def n_unique(
        self,
        key: str,
        default: Any | Callable[[I], Any] = MISSING,
    ) -> int:
        """Count the number of unique values for a specific key across all items.

        Args:
            key: The key to count unique values for.
            default: The default value to return if the key is not found.
                If a callable, it will be called with the item
                and the value returned will be used as the default.

        Returns:
            int: The number of unique values for the specified key.

        """
        return len(self.unique(key, default))

    def sort(self, *keys: str, reverse: bool = False) -> Self:
        """Sort items based on one or more keys.

        Args:
            *keys: The keys to sort by, in order of priority.
            reverse: Whether to sort in descending order (default is
                ascending).

        Returns:
            Self: A new Collection with the items sorted according to
            the specified keys.

        """
        if not keys:
            return self

        arrays = [self.to_numpy(key) for key in keys]
        index = np.lexsort(arrays[::-1])

        if reverse:
            index = index[::-1]

        return self[index]

    def map[**P, R](
        self,
        function: Callable[Concatenate[I, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iterator[R]:
        """Apply a function to each item and return an iterator of results.

        This is a memory-efficient mapping operation that lazily evaluates results.
        Ideal for large collections where memory usage is a concern.

        Args:
            function: Function to apply to each item. The item is passed
                as the first argument.
            *args: Additional positional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            Iterator[R]: An iterator of the function's results.

        Examples:
            ```python
            # Process results one at a time
            for result in collection.map(process_item, additional_arg):
                handle_result(result)

            # Convert to list if needed
            results = list(collection.map(transform_item))
            ```

        """
        yield from (function(i, *args, **kwargs) for i in self)

    def pmap[**P, R](
        self,
        function: Callable[Concatenate[I, P], R],
        n_jobs: int = -1,
        backend: str = "multiprocessing",
        progress: bool = False,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> list[R]:
        """Apply a function to each item in parallel and return a list of results.

        This method processes items concurrently for improved performance on
        CPU-bound or I/O-bound operations, depending on the backend.

        Args:
            function: Function to apply to each item. The item is passed
                as the first argument.
            n_jobs (int): Number of jobs to run in parallel. -1 means using all
                processors.
            backend (str): Parallelization backend.
            progress (bool): Whether to display a progress bar.
            *args: Additional positional arguments to pass to the function.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            list[R]: A list containing all results of the function applications.

        Examples:
            ```python
            # Process all items in parallel using all cores
            results = collection.pmap(heavy_computation)

            # Specify number of parallel jobs and backend
            results = collection.pmap(process_files, n_jobs=4, backend="threading")
            ```

        """
        parallel = Parallel(n_jobs=n_jobs, backend=backend, return_as="list")
        it = (delayed(function)(i, *args, **kwargs) for i in self)

        if not progress:
            return parallel(it)  # type: ignore

        from hydraflow.utils.progress import Progress

        with Progress(*Progress.get_default_columns()) as p:
            p.add_task("", total=len(self))
            return parallel(it)  # type: ignore

    def to_frame(
        self,
        *keys: str | tuple[str, Any | Callable[[I], Any]],
        defaults: dict[str, Any | Callable[[I], Any]] | None = None,
        n_jobs: int = 0,
        backend: str = "multiprocessing",
        progress: bool = False,
        **kwargs: Callable[[I], Any],
    ) -> DataFrame:
        """Convert the collection to a Polars DataFrame.

        This method converts the items in the collection into a Polars DataFrame.
        It allows specifying multiple keys, where each key can be a string or a tuple.
        If a tuple is provided, the first element is treated as the key and the second
        element as the default value for that key.

        Args:
            *keys (str | tuple[str, Any | Callable[[I], Any]]): The keys to include
                as columns in the DataFrame. If a tuple is provided, the first element
                is the key and the second element is the default value.
            defaults (dict[str, Any | Callable[[I], Any]] | None): Default values
                for the keys. If a callable, it will be called with the item and the
                value returned will be used as the default.
            n_jobs (int): Number of jobs to run in parallel. 0 means no parallelization.
                Default to 0.
            backend (str): Parallelization backend.
            progress (bool): Whether to display a progress bar.
            **kwargs (Callable[[I], Any]): Additional columns to compute using
                callables that take an item and return a value.

        Returns:
            DataFrame: A Polars DataFrame containing the specified data from the items.

        Examples:
            ```python
            # Convert to DataFrame with single keys
            df = collection.to_frame("name", "age")

            # Convert to DataFrame with keys and default values
            df = collection.to_frame(("name", "Unknown"), ("age", 0))
            ```

        """
        if defaults is None:
            defaults = {}

        keys_ = []
        for k in keys:
            if isinstance(k, tuple):
                keys_.append(k[0])
                defaults[k[0]] = k[1]
            else:
                keys_.append(k)

        data = {k: self.to_list(k, defaults.get(k, MISSING)) for k in keys_}
        df = DataFrame(data)

        if not kwargs:
            return df

        kv = kwargs.items()
        if n_jobs == 0:
            return df.with_columns(Series(k, self.map(v)) for k, v in kv)

        columns = [Series(k, self.pmap(v, n_jobs, backend, progress)) for k, v in kv]
        return df.with_columns(*columns)

    def group_by(self, *by: str) -> GroupBy[Self, I]:
        """Group items by one or more keys and return a GroupBy instance.

        This method organizes items into groups based on the specified
        keys and returns a GroupBy instance that contains the grouped
        collections. The GroupBy instance behaves like a dictionary,
        allowing access to collections for each group key.

        Args:
            *by: The keys to group by. If a single key is provided,
                its value will be used as the group key.
                If multiple keys are provided, a tuple of their
                values will be used as the group key.
                Keys can use dot notation (e.g., "model.type")
                to access nested configuration values.

        Returns:
            GroupBy[Self, I]: A GroupBy instance containing the grouped items.
            Each group is a collection of the same type as the original.

        """
        groups: dict[Any, Self] = {}

        for item in self:
            keys = [to_hashable(self._get(item, key, MISSING)) for key in by]
            key = keys[0] if len(by) == 1 else tuple(keys)

            if key not in groups:
                groups[key] = self.__class__([], self._get)

            groups[key]._items.append(item)  # noqa: SLF001

        return GroupBy(by, groups)

    def sample(self, k: int, seed: int | None = None) -> Self:
        """Sample a random subset of items from the collection.

        This method returns a new collection containing a random sample
        of items from the original collection. The sample is drawn without
        replacement, meaning each item can only appear once in the sample.

        Args:
            k (int): The number of items to sample.
            seed (int | None): The seed for the random number generator.
                If provided, the sample will be reproducible.

        Returns:
            Self: A new collection containing a random sample of items.

        Raises:
            ValueError: If the sample size is greater than the collection size.

        """
        n = len(self)
        if k < 1 or k > n:
            msg = f"Sample size ({k}) must be between 1 and {n}"
            raise ValueError(msg)

        if seed is not None:
            random.seed(seed)

        return self.__class__(random.sample(self._items, k), self._get)

    def shuffle(self, seed: int | None = None) -> Self:
        """Shuffle the items in the collection.

        This method returns a new collection with the items in random order.

        Args:
            seed (int | None): The seed for the random number generator.
                If provided, the sample will be reproducible.

        Returns:
            Self: A new collection containing the items in random order.

        """
        return self.sample(len(self), seed)

    def eq(
        self,
        left: str,
        right: str,
        *,
        default: Any | Callable[[I], Any] = MISSING,
    ) -> Callable[[I], bool]:
        """Create a predicate function that checks if two attributes are equal.

        Args:
            left (str): The name of the left attribute to compare.
            right (str): The name of the right attribute to compare.
            default (Any | Callable[[I], Any], optional): The default value
                to use if either attribute is not found. If callable, it
                will be called with the item.

        Returns:
            Callable[[I], bool]: A function that takes an item and returns
            True if the values of the specified attributes are equal.

        Examples:
            ```python
            # Find items where attribute 'a' equals attribute 'b'
            equal_items = collection.filter(collection.eq('a', 'b'))
            ```

        """
        return lambda i: self._get(i, left, default) == self._get(i, right, default)

    def ne(
        self,
        left: str,
        right: str,
        *,
        default: Any | Callable[[I], Any] = MISSING,
    ) -> Callable[[I], bool]:
        """Create a predicate function that checks if two attributes are not equal.

        Args:
            left (str): The name of the left attribute to compare.
            right (str): The name of the right attribute to compare.
            default (Any | Callable[[I], Any], optional): The default value
                to use if either attribute is not found. If callable, it
                will be called with the item.

        Returns:
            Callable[[I], bool]: A function that takes an item and returns
            True if the values of the specified attributes are not equal.

        Examples:
            ```python
            # Find items where attribute 'a' is not equal to attribute 'b'
            unequal_items = collection.filter(collection.ne('a', 'b'))
            ```

        """
        return lambda i: self._get(i, left, default) != self._get(i, right, default)

    def gt(
        self,
        left: str,
        right: str,
        *,
        default: Any | Callable[[I], Any] = MISSING,
    ) -> Callable[[I], bool]:
        """Create a predicate function that checks if the left > the right.

        Args:
            left (str): The name of the left attribute to compare.
            right (str): The name of the right attribute to compare.
            default (Any | Callable[[I], Any], optional): The default value
                to use if either attribute is not found. If callable, it
                will be called with the item.

        Returns:
            Callable[[I], bool]: A function that takes an item and returns
            True if the left attribute value is greater than the right attribute value.

        Examples:
            ```python
            # Find items where attribute 'a' is greater than attribute 'b'
            items = collection.filter(collection.gt('a', 'b'))
            ```

        """
        return lambda i: self._get(i, left, default) > self._get(i, right, default)

    def lt(
        self,
        left: str,
        right: str,
        *,
        default: Any | Callable[[I], Any] = MISSING,
    ) -> Callable[[I], bool]:
        """Create a predicate function that checks if the left < the right.

        Args:
            left (str): The name of the left attribute to compare.
            right (str): The name of the right attribute to compare.
            default (Any | Callable[[I], Any], optional): The default value
                to use if either attribute is not found. If callable, it
                will be called with the item.

        Returns:
            Callable[[I], bool]: A function that takes an item and returns
            True if the left attribute value is less than the right attribute value.

        Examples:
            ```python
            # Find items where attribute 'a' is less than attribute 'b'
            items = collection.filter(collection.lt('a', 'b'))
            ```

        """
        return lambda i: self._get(i, left, default) < self._get(i, right, default)

    def ge(
        self,
        left: str,
        right: str,
        *,
        default: Any | Callable[[I], Any] = MISSING,
    ) -> Callable[[I], bool]:
        """Create a predicate function that checks if the left >= the right.

        Args:
            left (str): The name of the left attribute to compare.
            right (str): The name of the right attribute to compare.
            default (Any | Callable[[I], Any], optional): The default value.

        Returns:
            Callable[[I], bool]: A predicate function for filtering.

        """
        return lambda i: self._get(i, left, default) >= self._get(i, right, default)

    def le(
        self,
        left: str,
        right: str,
        *,
        default: Any | Callable[[I], Any] = MISSING,
    ) -> Callable[[I], bool]:
        """Create a predicate function that checks if the left <= the right.

        Args:
            left (str): The name of the left attribute to compare.
            right (str): The name of the right attribute to compare.
            default (Any | Callable[[I], Any], optional): The default value.

        Returns:
            Callable[[I], bool]: A predicate function for filtering.

        """
        return lambda i: self._get(i, left, default) <= self._get(i, right, default)

    def startswith(
        self,
        key: str,
        prefix: str,
        *,
        default: Any | Callable[[I], Any] = MISSING,
    ) -> Callable[[I], bool]:
        """Create a predicate function that checks if an attribute starts with a prefix.

        Args:
            key (str): The name of the attribute to check.
            prefix (str): The prefix to check for.
            default (Any | Callable[[I], Any], optional): The default value.

        Returns:
            Callable[[I], bool]: A predicate function for filtering.

        """
        return lambda i: str(self._get(i, key, default)).startswith(prefix)

    def endswith(
        self,
        key: str,
        suffix: str,
        *,
        default: Any | Callable[[I], Any] = MISSING,
    ) -> Callable[[I], bool]:
        """Create a predicate function that checks if an attribute ends with a suffix.

        Args:
            key (str): The name of the attribute to check.
            suffix (str): The suffix to check for.
            default (Any | Callable[[I], Any], optional): The default value.

        Returns:
            Callable[[I], bool]: A predicate function for filtering.

        """
        return lambda i: str(self._get(i, key, default)).endswith(suffix)

    def match(
        self,
        key: str,
        pattern: str | re.Pattern,
        *,
        default: Any | Callable[[I], Any] = MISSING,
    ) -> Callable[[I], bool]:
        """Create a predicate function that checks if an attribute matches a pattern.

        Args:
            key (str): The name of the attribute to check.
            pattern (str | re.Pattern): The pattern to check for.
            default (Any | Callable[[I], Any], optional): The default value.

        Returns:
            Callable[[I], bool]: A predicate function for filtering.

        """
        return lambda i: re.match(pattern, str(self._get(i, key, default))) is not None


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
    msg = "No item found matching the specified criteria"
    return ValueError(msg)


def matches(value: Any, criterion: Any) -> bool:
    """Check if a value matches the given criterion.

    This function compares the value with the given criteria according
    to the following rules:

    - If criterion is callable: Call it with the value and return
      the boolean result
    - If criterion is a list or set: Check if the value is in the list/set
    - If criterion is a tuple of length 2: Check if the value is
      in the range [criterion[0], criterion[1]]. Both sides are
      inclusive
    - Otherwise: Check if the value equals the criteria

    Args:
        value: The value to be compared with the criterion.
        criterion: The criterion to match against. Can be:
            - A callable that takes the value and returns a boolean
            - A list or set to check membership
            - A tuple of length 2 to check range inclusion
            - Any other value for direct equality comparison

    Returns:
        bool: True if the value matches the criterion according to the rules above,
        False otherwise.

    Examples:
        >>> matches(5, lambda x: x > 3)
        True
        >>> matches(2, [1, 2, 3])
        True
        >>> matches(4, (1, 5))
        True
        >>> matches(3, 3)
        True

    """
    if callable(criterion):
        return bool(criterion(value))

    if isinstance(criterion, ListConfig):
        criterion = list(criterion)

    if isinstance(criterion, list | set) and not _is_iterable(value):
        return value in criterion

    if isinstance(criterion, tuple) and len(criterion) == 2 and not _is_iterable(value):
        return criterion[0] <= value <= criterion[1]

    if _is_iterable(criterion):
        criterion = list(criterion)

    if _is_iterable(value):
        value = list(value)

    return value == criterion


def _is_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, str)
