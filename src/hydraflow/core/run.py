from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from typing import TYPE_CHECKING, overload

from omegaconf import DictConfig, OmegaConf

from .run_info import RunInfo

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any, Self


class Run[C, I]:
    info: RunInfo
    impl_factory: Callable[[Path], I]

    def __init__(self, run_dir: Path, impl_factory: Callable[[Path], I]) -> None:
        self.info = RunInfo(run_dir)
        self.impl_factory = impl_factory

    @cached_property
    def cfg(self) -> C:
        config_file = self.info.run_dir / "artifacts/.hydra/config.yaml"
        return OmegaConf.load(config_file)  # type: ignore

    @cached_property
    def impl(self) -> I:
        return self.impl_factory(self.info.run_dir / "artifacts")

    @overload
    def set_default(
        self,
        key: str,
        value: Any | Callable[[Self], Any],
    ) -> None: ...

    @overload
    def set_default(
        self,
        key: tuple[str, ...],
        value: Iterable[Any] | Callable[[Self], Iterable[Any]],
    ) -> None: ...

    def set_default(
        self,
        key: str | tuple[str, ...],
        value: Any | Callable[[Self], Any],
    ) -> None:
        """Set default value(s) in the configuration if they don't already exist.

        This method adds a value or multiple values to the configuration, but only
        if the corresponding keys don't already have values. Existing values will not
        be modified.

        Args:
            key: Either a string representing a single configuration path (can use dot
                notation like "section.subsection.param"), or a tuple of strings to set
                multiple related configuration values at once.
            value: The value to set. This can be:
                - For string keys: Any value, or a callable that returns a value
                - For tuple keys: An iterable with the same length as the key tuple,
                or a callable that returns such an iterable

        Raises:
            TypeError: If a tuple key is provided but the value is not an iterable,
                or if the callable doesn't return an iterable.

        """
        _set_default(self, self.cfg, key, value)  # type: ignore


def _set_default(
    run: Any,
    cfg: DictConfig,
    key: str | tuple[str, ...],
    value: Any,
) -> None:
    if isinstance(key, str):
        if OmegaConf.select(cfg, key) is None:
            v = value(run) if callable(value) else value
            OmegaConf.update(cfg, key, v, force_add=True)
        return

    if all(OmegaConf.select(cfg, k) is not None for k in key):
        return

    if callable(value):
        value = value(run)

    if not isinstance(value, Iterable) or isinstance(value, str):
        msg = f"{value} is not an iterable"
        raise TypeError(msg)

    for k, v in zip(key, value, strict=True):
        if OmegaConf.select(cfg, k) is None:
            OmegaConf.update(cfg, k, v, force_add=True)
