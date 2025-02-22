"""Provide functionality for working with configuration objects using the OmegaConf."""

from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import DictConfig, ListConfig, OmegaConf

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any


def iter_params(config: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Recursively iterate over the parameters in the given configuration object.

    This function traverses the configuration object and yields key-value pairs
    representing the parameters. The keys are prefixed with the provided prefix.

    Args:
        config (Any): The configuration object to iterate over. This can be a
            dictionary, list, DictConfig, or ListConfig.
        prefix (str): The prefix to prepend to the parameter keys.
            Defaults to an empty string.

    Yields:
        Key-value pairs representing the parameters in the configuration object.

    """
    if config is None:
        return

    if isinstance(config, list) and all(isinstance(x, str) for x in config):
        config = _from_dotlist(config)

    if not isinstance(config, DictConfig | ListConfig):
        config = OmegaConf.create(config)

    yield from _iter_params(config, prefix)


def _from_dotlist(config: list[str]) -> dict[str, str]:
    result = {}
    for item in config:
        if "=" in item:
            key, value = item.split("=", 1)
            result[key.strip()] = value.strip()

    return result


def _iter_params(config: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(config, DictConfig):
        for key, value in config.items():
            if _is_param(value):
                yield f"{prefix}{key}", _convert(value)

            else:
                yield from _iter_params(value, f"{prefix}{key}.")

    elif isinstance(config, ListConfig):
        for index, value in enumerate(config):
            if _is_param(value):
                yield f"{prefix}{index}", _convert(value)

            else:
                yield from _iter_params(value, f"{prefix}{index}.")


def _is_param(value: Any) -> bool:
    """Check if the given value is a parameter."""
    if isinstance(value, DictConfig):
        return False

    if isinstance(value, ListConfig):
        if any(isinstance(v, DictConfig | ListConfig) for v in value):
            return False

    return True


def _convert(value: Any) -> Any:
    """Convert the given value to a Python object."""
    if isinstance(value, ListConfig):
        return list(value)

    return value


def select_config(config: Any, names: list[str]) -> dict[str, Any]:
    """Select the given parameters from the configuration object.

    This function selects the given parameters from the configuration object
    and returns a new configuration object containing only the selected parameters.

    Args:
        config (Any): The configuration object to select parameters from.
        names (list[str]): The names of the parameters to select.

    Returns:
        DictConfig: A new configuration object containing only the selected parameters.

    """
    if not isinstance(config, DictConfig):
        config = OmegaConf.structured(config)

    return {name: _get(config, name) for name in names}


def _get(config: DictConfig, name: str) -> Any:
    """Get the value of the given parameter from the configuration object."""
    if "." not in name:
        return config.get(name)

    prefix, name = name.split(".", 1)
    return _get(config.get(prefix), name)


def select_overrides(config: object, overrides: list[str]) -> dict[str, Any]:
    """Select the given overrides from the configuration object."""
    names = [override.split("=")[0].strip() for override in overrides]
    return select_config(config, names)
