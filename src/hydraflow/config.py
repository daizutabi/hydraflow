"""Provide functionality for working with configuration objects using the OmegaConf."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import DictConfig, ListConfig, OmegaConf

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any


def collect_params(config: object) -> dict[str, Any]:
    """Iterate over parameters and collect them into a dictionary.

    Args:
        config (object): The configuration object to iterate over.
        prefix (str): The prefix to prepend to the parameter keys.

    Returns:
        dict[str, Any]: A dictionary of collected parameters.

    """
    return dict(iter_params(config))


def iter_params(config: object, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Recursively iterate over the parameters in the given configuration object.

    This function traverses the configuration object and yields key-value pairs
    representing the parameters. The keys are prefixed with the provided prefix.

    Args:
        config (object): The configuration object to iterate over. This can be a
            dictionary, list, DictConfig, or ListConfig.
        prefix (str): The prefix to prepend to the parameter keys.
            Defaults to an empty string.

    Yields:
        Key-value pairs representing the parameters in the configuration object.

    """
    if config is None:
        return

    if not isinstance(config, DictConfig | ListConfig):
        config = OmegaConf.create(config)  # type: ignore

    yield from _iter_params(config, prefix)


def _iter_params(config: object, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(config, DictConfig):
        for key, value in config.items():
            if _is_param(value):
                yield f"{prefix}{key}", value

            else:
                yield from _iter_params(value, f"{prefix}{key}.")

    elif isinstance(config, ListConfig):
        for index, value in enumerate(config):
            if _is_param(value):
                yield f"{prefix}{index}", value

            else:
                yield from _iter_params(value, f"{prefix}{index}.")


def _is_param(value: object) -> bool:
    """Check if the given value is a parameter."""
    if isinstance(value, DictConfig):
        return False

    if isinstance(value, ListConfig):  # noqa: SIM102
        if any(isinstance(v, DictConfig | ListConfig) for v in value):
            return False

    return True
