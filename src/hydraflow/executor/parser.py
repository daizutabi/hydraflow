"""Parse and convert string representations of numbers and ranges.

This module provides utility functions for parsing and converting
string representations of numbers and ranges. It includes functions
to convert strings to numbers, count decimal places, handle numeric
ranges, and expand values from string arguments.
"""

from __future__ import annotations

import shlex
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


def count_decimal_digits(x: str) -> int:
    """Count decimal places in a string.

    Examine a string representing a number and returns the count
    of decimal places present after the decimal point.
    Return 0 if no decimal point is found.

    Args:
        x (str): The string to check.

    Returns:
        int: The number of decimal places.

    Examples:
        >>> count_decimal_digits("1")
        0
        >>> count_decimal_digits("-1.2")
        1
        >>> count_decimal_digits("1.234")
        3
        >>> count_decimal_digits("-1.234e-10")
        3

    """
    if "." not in x:
        return 0

    decimal_part = x.split(".")[1]
    if "e" in decimal_part.lower():
        decimal_part = decimal_part.split("e")[0]

    return len(decimal_part)


def count_integer_digits(num_str: str) -> int:
    """Count the number of digits in the integer part of a number.

    Consider only the integer part of a number, even if it is in
    scientific notation.

    Args:
        num_str (str): The string representing a number.

    Returns:
        int: The number of digits in the integer part of a number.
        (excluding the sign)

    Examples:
        >>> count_integer_digits("123")
        3
        >>> count_integer_digits("-123.45")
        3
        >>> count_integer_digits("+0.00123")
        1
        >>> count_integer_digits("-1.200")
        1
        >>> count_integer_digits("+1.20e3")
        1
        >>> count_integer_digits("-0.120e-3")
        1
        >>> count_integer_digits(".123")
        0

    """
    if num_str.startswith(("+", "-")):
        num_str = num_str[1:]

    if "e" in num_str.lower():
        num_str = num_str.lower().split("e")[0]

    if "." in num_str:
        int_part = num_str.split(".")[0]
        if not int_part:
            return 0
        return len(int_part)
    return len(num_str)


def is_number(x: str) -> bool:
    """Check if a string represents a valid number.

    Args:
        x (str): The string to check.

    Returns:
        bool: True if the string is a number, False otherwise.

    Examples:
        >>> is_number("1")
        True
        >>> is_number("-1.2")
        True
        >>> is_number("1.2.3")
        False

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


def _get_range(arg: str) -> tuple[float, float, float]:
    """Return a tuple of (start, stop, step)."""
    args = [to_number(x) for x in arg.split(":")]

    if len(args) == 2:
        if args[0] > args[1]:
            raise ValueError("start cannot be greater than stop")

        return (args[0], args[1], 1)

    if args[2] == 0:
        raise ValueError("step cannot be zero")
    if args[2] > 0 and args[0] > args[1]:
        raise ValueError("start cannot be greater than stop")
    if args[2] < 0 and args[0] < args[1]:
        raise ValueError("start cannot be less than stop")

    return args[0], args[1], args[2]


def _arange(start: float, stop: float, step: float) -> list[float]:
    """Generate a range of floating point numbers.

    This function generates a range of floating point numbers
    with protection against rounding errors.

    Args:
        start (float): The starting value.
        stop (float): The end value (inclusive).
        step (float): The step size.

    Returns:
        list[float]: A list of floating point numbers from start to stop
        (inclusive) with the given step.

    """
    if step == 0:
        raise ValueError("Step cannot be zero")

    epsilon = min(abs(start), abs(stop)) * 1e-5

    result = []
    current = start

    if step > 0:
        while current <= stop + epsilon:
            result.append(current)
            current += step
    else:
        while current >= stop - epsilon:
            result.append(current)
            current += step

    return result


def split_suffix(arg: str) -> tuple[str, str]:
    """Split a string into the prefix and suffix.

    The suffix is the part of the string that starts with a colon (:).
    The prefix is the part of the string that precedes the suffix.

    Args:
        arg (str): The string to split.

    Returns:
        tuple[str, str]: A tuple containing the prefix and suffix.

    Examples:
        >>> split_suffix("1:k")
        ('1', 'e3')
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

    Examples:
        >>> add_exponent("1", "e3")
        '1e3'
        >>> add_exponent("1", "")
        '1'
        >>> add_exponent("0", "e-3")
        '0'
        >>> add_exponent("0.0", "e-3")
        '0.0'

    """
    if value in ["0", "0.", "0.0"] or not exponent:
        return value

    return f"{value}{exponent}"


def split_parentheses(arg: str) -> Iterator[str]:
    """Split a string with parentheses into a list of strings.

    Args:
        arg (str): The string to split.

    Returns:
        Iterator[str]: An iterator of the split strings.

    Examples:
        >>> list(split_parentheses("a(b,c)m(e:f)k"))
        ['a', 'b,c', 'e-3', 'e:f', 'e3']
        >>> list(split_parentheses("(b,c)d(e:f)"))
        ['b,c', 'd', 'e:f']

    """
    current = ""

    for char in arg:
        if char in ("(", ")"):
            if current:
                yield SUFFIX_EXPONENT.get(current, current)
                current = ""
        else:
            current += char

    if current:
        yield SUFFIX_EXPONENT.get(current, current)


def collect_parentheses(arg: str) -> list[str]:
    """Collect values from a string with parentheses.

    Args:
        arg (str): The string to collect values from.

    Returns:
        list[str]: A list of the collected values.

    Examples:
        >>> collect_parentheses("(1:3,5:9:2,20)k")
        ['1e3', '2e3', '3e3', '5e3', '7e3', '9e3', '20e3']
        >>> collect_parentheses("2e(-1,-2,-3)")
        ['2e-1', '2e-2', '2e-3']
        >>> collect_parentheses("(1:3)e(3,5)")
        ['1e3', '2e3', '3e3', '1e5', '2e5', '3e5']

    """
    it = [expand_values(x) for x in split_parentheses(arg)]
    return ["".join(x[::-1]) for x in product(*it[::-1])]


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

    Examples:
        >>> collect_values("1:4")
        ['1', '2', '3', '4']
        >>> collect_values("1.2:1.4:0.1:k")
        ['1.2e3', '1.3e3', '1.4e3']
        >>> collect_values("0.1")
        ['0.1']
        >>> collect_values("4:M")
        ['4e6']
        >>> collect_values("(1:3,5:7)M")
        ['1e6', '2e6', '3e6', '5e6', '6e6', '7e6']
        >>> collect_values("(1,5)e-(1:3)")
        ['1e-1', '5e-1', '1e-2', '5e-2', '1e-3', '5e-3']

    """
    if "(" in arg:
        return collect_parentheses(arg)

    if ":" not in arg:
        return [arg]

    arg, exponent = split_suffix(arg)

    if ":" not in arg:
        return [f"{arg}{exponent}"]

    rng = _get_range(arg)

    if all(isinstance(x, int) for x in rng):
        values = [str(x) for x in _arange(*rng)]
    else:
        n = max(*(count_integer_digits(x) for x in arg.split(":")))
        m = max(*(count_decimal_digits(x) for x in arg.split(":")))
        values = [f"{x:.{n + m}g}" for x in _arange(*rng)]

    return [add_exponent(x, exponent) for x in values]


def split(arg: str) -> list[str]:
    """Split a string by top-level commas.

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
        >>> split("(a,b)m,(1,2:4)k")
        ['(a,b)m', '(1,2:4)k']

    """
    result = []
    current = []
    bracket_count = 0
    paren_count = 0
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
        elif char == "(" and not (in_single_quote or in_double_quote):
            paren_count += 1
        elif char == ")" and not (in_single_quote or in_double_quote):
            paren_count -= 1
        elif (
            char == ","
            and bracket_count == 0
            and paren_count == 0
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


def expand_values(arg: str, suffix: str = "") -> Iterator[str]:
    """Expand a string argument into a list of values.

    Take a string containing comma-separated values or ranges
    and return a list of all individual values.
    Handle numeric ranges and special characters.

    Args:
        arg (str): The argument to expand.
        suffix (str): The suffix to append to each value.

    Returns:
        Iterator[str]: An iterator of the expanded values.

    """
    suffix = SUFFIX_EXPONENT.get(suffix, suffix)

    for value in chain.from_iterable(collect_values(x) for x in split(arg)):
        yield f"{value}{suffix}"


def split_arg(arg: str) -> tuple[str, str, str]:
    """Split an argument into a key, suffix, and value.

    Args:
        arg (str): The argument to split.

    Returns:
        tuple[str, str, str]: A tuple containing the key,
        suffix, and value.

    """
    if "=" not in arg:
        msg = f"Invalid argument: {arg}"
        raise ValueError(msg)

    key, value = arg.split("=")

    if "/" in key:
        key, suffix = key.split("/")
        return key, suffix, value

    return key, "", value


def collect_arg(arg: str) -> str:
    """Collect a string of expanded key-value pairs.

    Take a key-value pair argument and concatenates all expanded
    values with commas, returning a single string suitable for
    command-line usage.

    Args:
        arg (str): The argument to collect.

    Returns:
        str: A string of the collected key and values.

    """
    key, suffix, value = split_arg(arg)
    value = ",".join(expand_values(value, suffix))
    return f"{key}={value}"


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
        key, suffix, value = split_arg(arg)

        for v in expand_values(value, suffix):
            yield f"{key}={v}"

        return

    args = arg.split("|")
    key = ""
    suffix = ""

    for arg_ in args:
        if "=" in arg_:
            key, suffix, value = split_arg(arg_)
        elif key:
            value = arg_
        else:
            msg = f"Invalid argument: {arg_}"
            raise ValueError(msg)

        value = ",".join(expand_values(value, suffix))
        yield f"{key}={value}"


def collect(args: str | list[str]) -> list[str]:
    """Collect a list of arguments into a list of strings.

    Args:
        args (list[str]): The arguments to collect.

    Returns:
        list[str]: A list of the collected arguments.

    """
    if isinstance(args, str):
        args = shlex.split(args)

    return [collect_arg(arg) for arg in args]


def expand(args: str | list[str]) -> list[list[str]]:
    """Expand a list of arguments into a list of lists of strings.

    Args:
        args (list[str]): The arguments to expand.

    Returns:
        list[list[str]]: A list of the expanded arguments.

    """
    if isinstance(args, str):
        args = shlex.split(args)

    return [list(x) for x in product(*(expand_arg(arg) for arg in args))]
