"""Parse and convert string representations of numbers and ranges.

This module provides utility functions for parsing and converting
string representations of numbers and ranges. It includes functions
to convert strings to numbers, count decimal places, handle numeric
ranges, and expand values from string arguments.
"""

from __future__ import annotations

from itertools import chain


def to_number(x: str) -> int | float:
    """Convert a string to an integer or float.

    Attempts to convert a string to an integer or a float,
    returning 0 if the string is empty or cannot be converted.

    Args:
        x (str): The string to convert.

    Returns:
        int | float: The converted number as an integer or float.

    """
    if not x:
        return 0

    if "." in x:
        return float(x)

    return int(x)


def num_point(x: str) -> int:
    """Count decimal places in a string.

    Examine a string representing a number and returns the count
    of decimal places present after the decimal point.
    Return 0 if no decimal point is found.

    Args:
        x (str): The string to check.

    Returns:
        int: The number of decimal places.

    """
    if "." not in x:
        return 0

    return len(x.split(".")[-1])


def _get_range(arg: str) -> tuple[float, float, float]:
    args = [to_number(x) for x in arg.split(":")]

    if len(args) == 2:
        if args[0] > args[1]:
            raise ValueError("start cannot be greater than stop")

        return (args[0], 1, args[1])

    if args[1] == 0:
        raise ValueError("step cannot be zero")
    if args[1] > 0 and args[0] > args[2]:
        raise ValueError("start cannot be greater than stop")
    if args[1] < 0 and args[0] < args[2]:
        raise ValueError("start cannot be less than stop")

    return args[0], args[1], args[2]


def _arange(start: float, step: float, stop: float) -> list[float]:
    result = []
    current = start

    while current <= stop if step > 0 else current >= stop:
        result.append(current)
        current += step

    return result


SUFFIX_EXPONENT = {
    "T": "e12",
    "G": "e9",
    "M": "e6",
    "k": "e3",
    "m": "e-3",
    "u": "e-6",
    "n": "e-9",
    "p": "e-12",
    "f": "e-15",
}


def split_suffix(arg: str) -> tuple[str, str]:
    """Split a string into a numeric range and a suffix.

    Args:
        arg (str): The string to split.

    Returns:
        tuple[str, str]: A tuple containing the numeric range and the suffix.

    """
    if ":" not in arg:
        return arg, ""

    rng, suffix = arg.rsplit(":", 1)
    if all(char.isdigit() or char in ".+-" for char in suffix):
        return arg, ""

    return rng, SUFFIX_EXPONENT.get(suffix, suffix)


def add_suffix(value: str, suffix: str) -> str:
    """Append a suffix to a value string.

    Args:
        value (str): The value to modify.
        suffix (str): The suffix to append.

    Returns:
        str: The value with the suffix added.

    """
    if value in ["0", "0.", "0.0"] or not suffix:
        return value

    return f"{value}{suffix}"


def collect_values(arg: str) -> list[str]:
    """Collect a list of values from a range argument.

    Collect all individual values within a numeric range
    represented by a string (e.g., `1:4`) and return them
    as a list of strings.
    Support both integer and floating-point ranges.

    Args:
        arg (str): The argument to collect.

    Returns:
        list[str]: A list of the collected values.

    """
    if ":" not in arg:
        return [arg]

    arg, suffix = split_suffix(arg)

    if ":" not in arg:
        return [add_suffix(arg, suffix)]

    rng = _get_range(arg)

    if all(isinstance(x, int) for x in rng):
        values = [str(x) for x in _arange(*rng)]  # type: ignore
    else:
        n = max(*(num_point(x) for x in arg.split(":")))
        values = [str(round(x, n)) for x in _arange(*rng)]

    return [add_suffix(x, suffix) for x in values]


def expand_values(arg: str) -> list[str]:
    """Expand a string argument into a list of values.

    Take a string containing comma-separated values or ranges and return a list
    of all individual values. Handle numeric ranges and special characters.

    Args:
        arg (str): The argument to expand.

    Returns:
        list[str]: A list of the expanded values.

    """
    return list(chain.from_iterable(collect_values(x) for x in arg.split(",")))


def parse(arg: str) -> str:
    """Parse a string argument into a comma-separated string.

    Args:
        arg (str): The argument to parse.

    Returns:
        str: A comma-separated string of the parsed values.

    """
    return ",".join(expand_values(arg))
