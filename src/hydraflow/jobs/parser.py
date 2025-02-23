"""Parse and convert string representations of numbers and ranges.

This module provides utility functions for parsing and converting
string representations of numbers and ranges. It includes functions
to convert strings to numbers, count decimal places, handle numeric
ranges, and expand values from string arguments.
"""

from __future__ import annotations

import re
from itertools import chain, product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


def to_number(x: str) -> int | float:
    """Convert a string to an integer or float.

    Attempts to convert a string to an integer or a float,
    returning 0 if the string is empty or cannot be converted.

    Args:
        x (str): The string to convert.

    Returns:
        int | float: The converted number as an integer or float.

    Examples:
        >>> type(to_number("1"))
        <class 'int'>
        >>> type(to_number("1.2"))
        <class 'float'>
        >>> to_number("")
        0
        >>> to_number("1e-3")
        0.001

    """
    if not x:
        return 0

    if "." in x or "e" in x.lower():
        return float(x)

    return int(x)


def count_decimal_places(x: str) -> int:
    """Count decimal places in a string.

    Examine a string representing a number and returns the count
    of decimal places present after the decimal point.
    Return 0 if no decimal point is found.

    Args:
        x (str): The string to check.

    Returns:
        int: The number of decimal places.

    Examples:
        >>> count_decimal_places("1")
        0
        >>> count_decimal_places("-1.2")
        1
        >>> count_decimal_places("1.234")
        3
        >>> count_decimal_places("-1.234e-10")
        3

    """
    if "." not in x:
        return 0

    decimal_part = x.split(".")[1]
    if "e" in decimal_part.lower():
        decimal_part = decimal_part.split("e")[0]

    return len(decimal_part)


def is_number(x: str) -> bool:
    """Check if a string is a number.

    Args:
        x (str): The string to check.

    Returns:
        bool: True if the string is a number, False otherwise.

    """
    try:
        float(x)
    except ValueError:
        return False
    return True


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


def convert_suffix_to_exponent(arg: str) -> str:
    """Convert engineering notation suffix to exponential notation.

    Args:
        arg (str): Input string with optional engineering suffix (e.g., '1k', '2M')

    Returns:
        str: String with suffix converted to exponential notation (e.g., '1e3', '2e6')

    Examples:
        >>> convert_suffix_to_exponent('1k')
        '1e3'
        >>> convert_suffix_to_exponent('-2.5M')
        '-2.5e6'
        >>> convert_suffix_to_exponent('invalid')
        'invalid'

    """
    if len(arg) < 2:
        return arg

    prefix, suffix = arg[:-1], arg[-1]
    if suffix not in SUFFIX_EXPONENT:
        return arg

    if is_number(prefix):
        return prefix + SUFFIX_EXPONENT[suffix]

    return arg


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


def split_suffix(arg: str) -> tuple[str, str]:
    """Split a string into prefix and suffix.

    Args:
        arg (str): The string to split.

    Returns:
        tuple[str, str]: A tuple containing the prefix and suffix.

    Examples:
        >>> split_suffix("1:2:k")
        ('1:2', 'e3')
        >>> split_suffix("1:2:M")
        ('1:2', 'e6')
        >>> split_suffix(":1:2:M")
        (':1:2', 'e6')

    """
    if len(arg) < 3 or ":" not in arg:
        return arg, ""

    prefix, suffix = arg.rsplit(":", 1)

    if suffix.lower().startswith("e"):
        return prefix, suffix

    if suffix not in SUFFIX_EXPONENT:
        return arg, ""

    return prefix, SUFFIX_EXPONENT[suffix]


def add_exponent(value: str, exponent: str) -> str:
    """Append an exponent to a value string.

    Args:
        value (str): The value to modify.
        exponent (str): The exponent to append.

    Returns:
        str: The value with the exponent added.

    """
    if value in ["0", "0.", "0.0"] or not exponent:
        return value

    return f"{value}{exponent}"


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
        return [convert_suffix_to_exponent(arg)]

    arg, exponent = split_suffix(arg)

    rng = _get_range(arg)

    if all(isinstance(x, int) for x in rng):
        values = [str(x) for x in _arange(*rng)]
    else:
        n = max(*(count_decimal_places(x) for x in arg.split(":")))
        values = [str(round(x, n)) for x in _arange(*rng)]

    return [add_exponent(x, exponent) for x in values]


def split(arg: str) -> list[str]:
    r"""Split a string by top-level commas.

    Splits a string by commas while respecting nested structures.
    Commas inside brackets and quotes are ignored, only splitting
    at the top-level commas.

    Args:
        arg (str): The string to split.

    Returns:
        list[str]: A list of split strings.

    Examples:
        >>> split("[a,1],[b,2]")
        ['[a,1]', '[b,2]']
        >>> split('"x,y",z')
        ['"x,y"', 'z']
        >>> split("'p,q',r")
        ["'p,q'", 'r']

    """
    result = []
    current = []
    bracket_count = 0
    in_single_quote = False
    in_double_quote = False

    for char in arg:
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif char == "[" and not (in_single_quote or in_double_quote):
            bracket_count += 1
        elif char == "]" and not (in_single_quote or in_double_quote):
            bracket_count -= 1
        elif (
            char == ","
            and bracket_count == 0
            and not in_single_quote
            and not in_double_quote
        ):
            result.append("".join(current))
            current = []
            continue
        current.append(char)

    if current:
        result.append("".join(current))

    return result


def expand_values(arg: str) -> list[str]:
    """Expand a string argument into a list of values.

    Take a string containing comma-separated values or ranges and return a list
    of all individual values. Handle numeric ranges and special characters.

    Args:
        arg (str): The argument to expand.

    Returns:
        list[str]: A list of the expanded values.

    """
    return list(chain.from_iterable(collect_values(x) for x in split(arg)))


def collect_arg(arg: str) -> str:
    """Collect a string of expanded key-value pairs.

    Take a key-value pair argument and concatenates all expanded values with commas,
    returning a single string suitable for command-line usage.

    Args:
        arg (str): The argument to collect.

    Returns:
        str: A string of the collected key and values.

    """
    key, arg = arg.split("=")
    arg = ",".join(expand_values(arg))
    return f"{key}={arg}"


def expand_arg(arg: str) -> Iterator[str]:
    """Parse a string argument into a list of values.

    Responsible for parsing a string that may contain multiple
    arguments separated by pipes ("|") and returns a list of all
    expanded arguments.

    Args:
        arg (str): The argument to parse.

    Returns:
       list[str]: A list of the parsed arguments.

    """
    if "|" not in arg:
        key, value = arg.split("=")

        for v in expand_values(value):
            yield f"{key}={v}"

        return

    args = arg.split("|")
    key = ""

    for arg_ in args:
        if "=" in arg_:
            key, value = arg_.split("=")
        elif key:
            value = arg_
        else:
            msg = f"Invalid argument: {arg_}"
            raise ValueError(msg)

        value = ",".join(expand_values(value))
        yield f"{key}={value}"


def collect(args: str | list[str]) -> list[str]:
    """Collect a list of arguments into a list of strings.

    Args:
        args (list[str]): The arguments to collect.

    Returns:
        list[str]: A list of the collected arguments.

    """
    if isinstance(args, str):
        args = re.split(r"\s+", args.strip())

    args = [arg for arg in args if "=" in arg]

    return [collect_arg(arg) for arg in args]


def expand(args: str | list[str]) -> list[list[str]]:
    """Expand a list of arguments into a list of lists of strings.

    Args:
        args (list[str]): The arguments to expand.

    Returns:
        list[list[str]]: A list of the expanded arguments.

    """
    if isinstance(args, str):
        args = re.split(r"\s+", args.strip())

    args = [arg for arg in args if "=" in arg]

    return [list(x) for x in product(*(expand_arg(arg) for arg in args))]
