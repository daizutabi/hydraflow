from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import DictConfig, ListConfig, OmegaConf

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any


def iter_params(config: object, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if not isinstance(config, DictConfig | ListConfig):
        config = OmegaConf.create(config)  # type: ignore

    if isinstance(config, DictConfig):
        for key, value in config.items():
            if isinstance(value, (DictConfig, ListConfig)):
                yield from iter_params(value, f"{prefix}{key}.")

            else:
                yield f"{prefix}{key}", value

    elif isinstance(config, ListConfig):
        for index, value in enumerate(config):
            if isinstance(value, (DictConfig, ListConfig)):
                yield from iter_params(value, f"{prefix}{index}.")

            else:
                yield f"{prefix}{index}", value
