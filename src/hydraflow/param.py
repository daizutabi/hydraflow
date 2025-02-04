"""Provide utility functions for parameter matching.

The main function `match` checks if a given parameter matches a specified value.
It supports various types of values including None, boolean, list, tuple, int,
float, and str.

Helper functions `_match_list` and `_match_tuple` are used internally to handle
matching for list and tuple types respectively.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omegaconf import ListConfig, OmegaConf

if TYPE_CHECKING:
    from mlflow.entities import Run


def match(param: str, value: Any) -> bool:  # noqa: PLR0911
    """Check if the string matches the specified value.

    Args:
        param (str): The parameter to check.
        value (Any): The value to check.

    Returns:
        True if the parameter matches the specified value,
        False otherwise.

    """
    if callable(value):
        return value(param)

    if any(value is x for x in [None, True, False]):
        return param == str(value)

    if isinstance(value, list) and (m := _match_list(param, value)) is not None:
        return m

    if isinstance(value, tuple) and (m := _match_tuple(param, value)) is not None:
        return m

    if isinstance(value, int | float):
        return float(param) == value

    if isinstance(value, str):
        return param == value

    return param == str(value)


def _match_list(param: str, value: list) -> bool | None:
    if not value:
        return None

    if any(param.startswith(x) for x in ["[", "(", "{"]):
        return None

    if isinstance(value[0], bool):
        return None

    if not isinstance(value[0], int | float | str):
        return None

    return type(value[0])(param) in value


def _match_tuple(param: str, value: tuple) -> bool | None:
    if len(value) != 2:  # noqa: PLR2004
        return None

    if any(param.startswith(x) for x in ["[", "(", "{"]):
        return None

    if isinstance(value[0], bool):
        return None

    if not isinstance(value[0], int | float | str):
        return None

    if type(value[0]) is not type(value[1]):
        return None

    return value[0] <= type(value[0])(param) <= value[1]  # type: ignore


def to_value(param: str | None, type_: type) -> Any:
    """Convert the parameter to the specified type.

    Args:
        param (str | None): The parameter to convert.
        type_ (type): The type to convert to.

    Returns:
        The converted value.

    """
    if param is None or param == "None":
        return None

    if type_ is int:
        return int(param)

    if type_ is float:
        return float(param)

    if type_ is bool:
        return param == "True"

    if type_ is list or type_ is ListConfig:
        return list(OmegaConf.create(param))

    return param


def get_params(run: Run, *names: str | list[str]) -> tuple[str | None, ...]:
    """Retrieve the values of specified parameters from the given run.

    This function extracts the values of the parameters identified by the
    provided names from the specified run. It can accept both individual
    parameter names and lists of parameter names.

    Args:
        run (Run): The run object from which to extract parameter values.
        *names (str | list[str]): The names of the parameters to retrieve.
            This can be a single parameter name or multiple names provided
            as separate arguments or as a list.

    Returns:
        tuple[str | None, ...]: A tuple containing the values of the specified
        parameters in the order they were provided.

    """
    names_ = []
    for name in names:
        if isinstance(name, list):
            names_.extend(name)
        else:
            names_.append(name)

    params = run.data.params
    return tuple(params.get(name) for name in names_)


def get_values(run: Run, names: list[str], types: list[type]) -> tuple[Any, ...]:
    """Retrieve the values of specified parameters from the given run.

    This function extracts the values of the parameters identified by the
    provided names from the specified run.

    Args:
        run (Run): The run object from which to extract parameter values.
        names (list[str]): The names of the parameters to retrieve.
        types (list[type]): The types to convert to.

    Returns:
        tuple[Any, ...]: A tuple containing the values of the specified
        parameters in the order they were provided.

    """
    params = get_params(run, names)
    it = zip(params, types, strict=True)
    return tuple(to_value(param, type_) for param, type_ in it)
