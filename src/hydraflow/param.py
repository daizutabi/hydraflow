"""Provide utility functions for parameter matching.

The main function `match` checks if a given parameter matches a specified value.
It supports various types of values including None, boolean, list, tuple, int,
float, and str.

Helper functions `_match_list` and `_match_tuple` are used internally to handle
matching for list and tuple types respectively.
"""

from __future__ import annotations

from typing import Any


def match(param: str, value: Any) -> bool:
    """Check if the string matches the specified value.

    Args:
        param (str): The parameter to check.
        value (Any): The value to check.

    Returns:
        True if the parameter matches the specified value,
        False otherwise.

    """
    if value in [None, True, False]:
        return param == str(value)

    if isinstance(value, list) and (m := _match_list(param, value)) is not None:
        return m

    if isinstance(value, tuple) and (m := _match_tuple(param, value)) is not None:
        return m

    if isinstance(value, str):
        return param == value

    if isinstance(value, int | float):
        return type(value)(param) == value

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
