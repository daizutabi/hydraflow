from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, overload

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any, Self


class Run[C]:
    run_dir: Path
    job_name: str
    cfg: C

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.job_name = get_job_name(run_dir)
        config_file = run_dir / "artifacts/.hydra/config.yaml"
        self.cfg = OmegaConf.load(config_file)  # type: ignore

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

    @cached_property
    def config_file(self) -> Path:
        return self.run_dir / "artifacts/.hydra/config.yaml"

    def load_config(self) -> DictConfig:
        return OmegaConf.load(self.config_file)  # type: ignore
