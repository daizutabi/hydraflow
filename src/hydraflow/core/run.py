"""Run module for HydraFlow.

This module provides the Run class, which represents an MLflow
Run in HydraFlow. A Run contains three main components:

1. info: Information about the run, such as run directory,
   run ID, and job name.
2. cfg: Configuration loaded from the Hydra configuration file.
3. impl: Implementation object created by the provided
   factory function.

The Run class allows accessing these components through
a unified interface, and provides methods for setting default
configuration values and filtering runs.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from typing import TYPE_CHECKING, overload

from omegaconf import DictConfig, ListConfig, OmegaConf

from .run_info import RunInfo

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any, Self


class Run[C, I = None]:
    """Represent an MLflow Run in HydraFlow.

    A Run contains information about the run, configuration, and
    implementation. The configuration type C and implementation
    type I are specified as type parameters.
    """

    info: RunInfo
    """Information about the run, such as run directory, run ID, and job name."""

    impl_factory: Callable[[Path], I]
    """Factory function to create the implementation object."""

    def __init__(
        self,
        run_dir: Path,
        impl_factory: Callable[[Path], I] = lambda _: None,
    ) -> None:
        self.info = RunInfo(run_dir)
        self.impl_factory = impl_factory

    @cached_property
    def cfg(self) -> C:
        """The configuration object loaded from the Hydra configuration file."""
        config_file = self.info.run_dir / "artifacts/.hydra/config.yaml"
        if config_file.exists():
            return OmegaConf.load(config_file)  # type: ignore

        return OmegaConf.create()  # type: ignore

    @cached_property
    def impl(self) -> I:
        """The implementation object created by the factory function."""
        return self.impl_factory(self.info.run_dir / "artifacts")

    @overload
    def update(
        self,
        key: str,
        value: Any | Callable[[Self], Any],
        *,
        force: bool = False,
    ) -> None: ...

    @overload
    def update(
        self,
        key: tuple[str, ...],
        value: Iterable[Any] | Callable[[Self], Iterable[Any]],
        *,
        force: bool = False,
    ) -> None: ...

    def update(
        self,
        key: str | tuple[str, ...],
        value: Any | Callable[[Self], Any],
        *,
        force: bool = False,
    ) -> None:
        """Set default value(s) in the configuration if they don't already exist.

        This method adds a value or multiple values to the configuration,
        but only if the corresponding keys don't already have values.
        Existing values will not be modified.

        Args:
            key: Either a string representing a single configuration path
                (can use dot notation like "section.subsection.param"),
                or a tuple of strings to set multiple related configuration
                values at once.
            value: The value to set. This can be:
                - For string keys: Any value, or a callable that returns
                  a value
                - For tuple keys: An iterable with the same length as the
                  key tuple, or a callable that returns such an iterable
                - For callable values: The callable must accept a single argument
                  of type Run (self) and return the appropriate value type
            force: Whether to force the update even if the key already exists.

        Raises:
            TypeError: If a tuple key is provided but the value is
                not an iterable, or if the callable doesn't return
                an iterable.

        """
        cfg: DictConfig = self.cfg  # type: ignore

        if isinstance(key, str):
            if force or OmegaConf.select(cfg, key) is None:
                v = value(self) if callable(value) else value  # type: ignore
                OmegaConf.update(cfg, key, v, force_add=True)
            return

        if not force and all(OmegaConf.select(cfg, k) is not None for k in key):
            return

        if callable(value):
            value = value(self)  # type: ignore

        if not isinstance(value, Iterable) or isinstance(value, str):
            msg = f"{value} is not an iterable"
            raise TypeError(msg)

        for k, v in zip(key, value, strict=True):
            if force or OmegaConf.select(cfg, k) is None:
                OmegaConf.update(cfg, k, v, force_add=True)

    def try_get(self, key: str) -> Any | None:
        """Try to get a value from the implementation, configuration, or run information.

        The method looks for the key in the following order:
        1. Implementation object (if available)
        2. Configuration object
        3. Run information (if key starts with "info.")

        Args:
            key: The key to look for. Can use dot notation for nested keys
                in configuration, or "info." prefix for run information.

        Returns:
            Any | None: The value associated with the key, or None if the key
            is not found.

        """  # noqa: E501
        if self.impl is not None and hasattr(self.impl, key):
            return getattr(self.impl, key)

        value = OmegaConf.select(self.cfg, key)  # type: ignore
        if value is not None:
            return value

        if key.startswith("info."):
            key = key[len("info.") :]
            if hasattr(self.info, key):
                return getattr(self.info, key)

        return None

    def get(self, key: str) -> Any:
        """Get a value from the implementation, configuration, or run information.

        The method looks for the key in the following order:
        1. Implementation object (if available)
        2. Configuration object
        3. Run information (if key starts with "info.")

        Args:
            key: The key to look for. Can use dot notation for nested keys
                in configuration, or "info." prefix for run information.

        Returns:
            Any: The value associated with the key.

        Raises:
            AttributeError: If the key is not found in any of the components.

        """
        value = self.try_get(key)
        if value is not None:
            return value

        msg = f"Key not found: {key}"
        raise AttributeError(msg)

    def predicate(self, key: str, value: Any) -> bool:
        """Check if a value satisfies a condition for filtering.

        This method retrieves the attribute specified by the key
        using the get method, and then compares it with the given
        value according to the following rules:

        - If value is callable: Call it with the attribute and return
          the boolean result
        - If value is a list or set: Check if the attribute is in the list/set
        - If value is a tuple of length 2: Check if the attribute is
          in the range [value[0], value[1]]. Both sides are inclusive
        - Otherwise: Check if the attribute equals the value

        Args:
            key: The key to get the attribute from.
            value: The value to compare with, or a callable that takes
                the attribute and returns a boolean.

        Returns:
            bool: True if the attribute satisfies the condition, False otherwise.

        """
        attr = self.get(key)

        if callable(value):
            return bool(value(attr))

        if isinstance(value, ListConfig):
            value = list(value)

        if isinstance(value, list | set) and not isinstance(attr, list | set):
            return attr in value

        if isinstance(value, tuple) and len(value) == 2 and not isinstance(attr, tuple):
            return value[0] <= attr <= value[1]

        return attr == value
