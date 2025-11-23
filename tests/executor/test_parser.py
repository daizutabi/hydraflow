from __future__ import annotations

import numpy as np
import pytest

from hydraflow.executor.parser import (
    _arange,
    _get_range,
    collect,
    collect_arg,
    collect_values,
    count_decimal_digits,
    count_integer_digits,
    expand,
    expand_arg,
    expand_values,
    split_arg,
    split_suffix,
    to_number,
)

# pyright: reportPrivateUsage=false


@pytest.mark.parametrize(("s", "x"), [("1", 1), ("1.2", 1.2), ("", 0)])
def test_to_number(s: str, x: float):
    assert to_number(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [("1", 0), ("1.2", 1), ("1.234", 3), ("123.", 0), ("", 0), ("1.234e-10", 3)],
)
def test_count_decimal_digits(s: str, x: int):
    assert count_decimal_digits(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [("1", 1), ("1.2", 1), ("1.234", 1), ("123.", 3), ("", 0), ("1.234e-10", 1)],
)
def test_count_integer_digits(s: str, x: int):
    assert count_integer_digits(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [
        ("1:2", (1, 2, 1)),
        (":2", (0, 2, 1)),
        ("0.1:0.4:0.1", (0.1, 0.4, 0.1)),
        ("1.2:1.4:0.1", (1.2, 1.4, 0.1)),
    ],
)
def test_get_range(s: str, x: tuple[float, float, float]):
    assert _get_range(s) == x


@pytest.mark.parametrize(
    ("arg", "expected_exception", "expected_message"),
    [
        ("5:3", ValueError, "start cannot be greater than stop"),
        ("1:2:0", ValueError, "step cannot be zero"),
        ("3:5:-1", ValueError, "start cannot be less than stop"),
        ("4.5:3.5:1.0", ValueError, "start cannot be greater than stop"),
    ],
)
def test_get_range_errors(
    arg: str,
    expected_exception: type[BaseException],
    expected_message: str,
):
    with pytest.raises(expected_exception) as excinfo:
        _get_range(arg)
    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    ("start", "stop", "step", "expected"),
    [
        (1.0, 5.0, 2.0, [1.0, 3.0, 5.0]),
        (1.2, 1.4, 0.1, [1.2, 1.3, 1.4]),
        (1.2e-12, 1.4e-12, 0.1e-12, [1.2e-12, 1.3e-12, 1.4e-12]),
        (1.4, 1.2, -0.1, [1.4, 1.3, 1.2]),
        (1.02e-3, 1.04e-3, 0.01e-3, [0.00102, 0.00103, 0.00104]),
    ],
)
def test_arange(start: float, stop: float, step: float, expected: list[float]):
    np.testing.assert_allclose(_arange(start, stop, step), expected)


def test_arange_error():
    with pytest.raises(ValueError):
        _arange(1.0, 1.0, 0.0)


@pytest.mark.parametrize(
    ("s", "x"),
    [
        ("1:2:3:suffix", ("1:2:3:suffix", "")),
        ("1:2:3", ("1:2:3", "")),
        ("1.23", ("1.23", "")),
        ("1:2:3:M", ("1:2:3", "e6")),
        ("1:2:3:k", ("1:2:3", "e3")),
        ("1:2:3:m", ("1:2:3", "e-3")),
        ("1:2:3:n", ("1:2:3", "e-9")),
        ("1:2:3:p", ("1:2:3", "e-12")),
        ("1:k", ("1", "e3")),
        ("1:2:k", ("1:2", "e3")),
        ("1:2:M", ("1:2", "e6")),
        (":1:2:M", (":1:2", "e6")),
        ("1:2:3:e-3", ("1:2:3", "e-3")),
        ("1:2:3:E8", ("1:2:3", "E8")),
        ("", ("", "")),
        ("1", ("1", "")),
        ("ab", ("ab", "")),
    ],
)
def test_split_suffix(s: str, x: tuple[str, str]):
    assert split_suffix(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [
        ("1", ["1"]),
        ("1k", ["1k"]),
        ("1:m", ["1e-3"]),
        ("1:M", ["1e6"]),
        ("0.234p", ["0.234p"]),
        ("1:3", ["1", "2", "3"]),
        ("0:1:0.25", ["0", "0.25", "0.5", "0.75", "1"]),
        ("10:11:0.25", ["10", "10.25", "10.5", "10.75", "11"]),
        (":3", ["0", "1", "2", "3"]),
        ("5:7", ["5", "6", "7"]),
        ("-1:1", ["-1", "0", "1"]),
        ("1:2:0.5", ["1", "1.5", "2"]),
        ("1.:2:0.5", ["1", "1.5", "2"]),
        ("2:3:0.5", ["2", "2.5", "3"]),
        ("-1:1:0.5", ["-1", "-0.5", "0", "0.5", "1"]),
        ("4:2:-1", ["4", "3", "2"]),
        ("4.5:2:-1.5", ["4.5", "3"]),
        ("4.5:1.5:-1.5", ["4.5", "3", "1.5"]),
        ("4.5:-4.5:-1.5", ["4.5", "3", "1.5", "0", "-1.5", "-3", "-4.5"]),
        ("1:2:u", ["1e-6", "2e-6"]),
        ("1:2:.25:n", ["1e-9", "1.25e-9", "1.5e-9", "1.75e-9", "2e-9"]),
        ("1.2:1.4:0.1:k", ["1.2e3", "1.3e3", "1.4e3"]),
        ("1.2e-3:1.4e-3:0.1e-3", ["0.0012", "0.0013", "0.0014"]),
        ("1.2e-12:1.4e-12:0.1e-12", ["1.2e-12", "1.3e-12", "1.4e-12"]),
        ("1.22e-6:1.24e-6:0.01e-6", ["1.22e-06", "1.23e-06", "1.24e-06"]),
        ("1.2e3:1.4e3:0.1e3", ["1.2e+03", "1.3e+03", "1.4e+03"]),
        ("1.2e6:1.4e6:0.1e6", ["1.2e+06", "1.3e+06", "1.4e+06"]),
        ("1:2:e2", ["1e2", "2e2"]),
        (":2:e2", ["0", "1e2", "2e2"]),
        ("-2:2:k", ["-2e3", "-1e3", "0", "1e3", "2e3"]),
        ("(1:3,5:9:2,20)k", ["1e3", "2e3", "3e3", "5e3", "7e3", "9e3", "20e3"]),
    ],
)
def test_collect_value(s: str, x: list[str]):
    assert collect_values(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [
        ("1,2,3", ["1", "2", "3"]),
        ("1:3,5:6", ["1", "2", "3", "5", "6"]),
        ("0:1:0.25,2.0", ["0", "0.25", "0.5", "0.75", "1", "2.0"]),
        ("3", ["3"]),
        ("3:k", ["3e3"]),
        ("1:3:k,3:7:2:M", ["1e3", "2e3", "3e3", "3e6", "5e6", "7e6"]),
        ("1:M,3:7:2:M", ["1e6", "3e6", "5e6", "7e6"]),
        ("[1,2],[3,4]", ["[1,2]", "[3,4]"]),
        ("'1,2','3,4'", ["'1,2'", "'3,4'"]),
        ('"1,2","3,4"', ['"1,2"', '"3,4"']),
        ("(1,4)k,(6,8)M", ["1e3", "4e3", "6e6", "8e6"]),
        ("(1:3)e-2,(5:7)e-3", ["1e-2", "2e-2", "3e-2", "5e-3", "6e-3", "7e-3"]),
    ],
)
def test_expand_value(s: str, x: list[str]):
    assert list(expand_values(s)) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [
        ("1,2,3", ["1e3", "2e3", "3e3"]),
        ("1:3,5:6", ["1e3", "2e3", "3e3", "5e3", "6e3"]),
        ("0:1:0.25,2.0", ["0e3", "0.25e3", "0.5e3", "0.75e3", "1e3", "2.0e3"]),
        ("3", ["3e3"]),
    ],
)
def test_expand_value_suffix(s: str, x: list[str]):
    assert list(expand_values(s, "k")) == x


def test_split_arg_error():
    with pytest.raises(ValueError):
        split_arg("1,2,3")


@pytest.mark.parametrize(
    ("s", "x"),
    [
        ("a=1", "a=1"),
        ("a/M=1", "a=1e6"),
        ("a=1,2", "a=1,2"),
        ("a/n=1,2", "a=1e-9,2e-9"),
        ("a=1:2", "a=1,2"),
        ("a/M=1:2", "a=1e6,2e6"),
        ("a=:3:2", "a=0,2"),
        ("a=(2,4)m,2,3", "a=2e-3,4e-3,2,3"),
        ("a=1:3:k", "a=1e3,2e3,3e3"),
        ("a=1:3:k,2:4:M", "a=1e3,2e3,3e3,2e6,3e6,4e6"),
        ("a/m=1:3,8:10", "a=1e-3,2e-3,3e-3,8e-3,9e-3,10e-3"),
    ],
)
def test_collect_arg(s: str, x: str):
    assert collect_arg(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [
        ("a=1", ["a=1"]),
        ("a=1,2", ["a=1", "a=2"]),
        ("a/M=1,2", ["a=1e6", "a=2e6"]),
        ("a=1:2", ["a=1", "a=2"]),
        ("a/n=1:2", ["a=1e-9", "a=2e-9"]),
        ("a=:3:2", ["a=0", "a=2"]),
        ("a=(0.1:0.4:0.1)k", ["a=0.1e3", "a=0.2e3", "a=0.3e3", "a=0.4e3"]),
        ("a=(1,2)(e-1,e2)", ["a=1e-1", "a=2e-1", "a=1e2", "a=2e2"]),
        ("a=(1,2)e(-1,2)", ["a=1e-1", "a=2e-1", "a=1e2", "a=2e2"]),
        ("a=1:3:k", ["a=1e3", "a=2e3", "a=3e3"]),
        ("a=1:3:k,2:4:M", ["a=1e3", "a=2e3", "a=3e3", "a=2e6", "a=3e6", "a=4e6"]),
        ("a=1,2|3,4", ["a=1,2", "a=3,4"]),
        ("a/G=1,2|3,4", ["a=1e9,2e9", "a=3e9,4e9"]),
        ("a=1:4|3:5:m", ["a=1,2,3,4", "a=3e-3,4e-3,5e-3"]),
        ("a=1,2|b=3,4|c=5,6", ["a=1,2", "b=3,4", "c=5,6"]),
        ("a/k=1,2|b/m=3,4|c/u=5,6", ["a=1e3,2e3", "b=3e-3,4e-3", "c=5e-6,6e-6"]),
    ],
)
def test_expand_arg(s: str, x: list[str]):
    assert list(expand_arg(s)) == x


def test_expand_arg_error():
    with pytest.raises(ValueError):
        list(expand_arg("1,2|3,4|"))


@pytest.mark.parametrize(
    ("s", "x"),
    [
        (["a=1"], ["a=1"]),
        (["a=1:3"], ["a=1,2,3"]),
        (["a/m=1:3"], ["a=1e-3,2e-3,3e-3"]),
        (["a=1:3", "b=4:6"], ["a=1,2,3", "b=4,5,6"]),
        (["a/k=1:3", "b/m=4:6"], ["a=1e3,2e3,3e3", "b=4e-3,5e-3,6e-3"]),
    ],
)
def test_collect_list(s: list[str], x: list[str]):
    assert collect(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [
        ("a=1:3\nb=4:6", ["a=1,2,3", "b=4,5,6"]),
        ("a/k=1:3 b=4:6", ["a=1e3,2e3,3e3", "b=4,5,6"]),
        ("a/n=4,5 b=c,d", ["a=4e-9,5e-9", "b=c,d"]),
        ("", []),
    ],
)
def test_collect_str(s: str, x: list[str]):
    assert collect(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [
        (["a=1"], [["a=1"]]),
        (["a/k=1,2"], [["a=1e3"], ["a=2e3"]]),
        (
            " a=1,2\n b=3,4\n",
            [["a=1", "b=3"], ["a=1", "b=4"], ["a=2", "b=3"], ["a=2", "b=4"]],
        ),
        (["a=1:2|3,4"], [["a=1,2"], ["a=3,4"]]),
        (["a/k=1:2|3,4"], [["a=1e3,2e3"], ["a=3e3,4e3"]]),
        (
            ["a=1:2|3,4", "b=5:6|c=7,8"],
            [
                ["a=1,2", "b=5,6"],
                ["a=1,2", "c=7,8"],
                ["a=3,4", "b=5,6"],
                ["a=3,4", "c=7,8"],
            ],
        ),
        (
            ["a/m=1:2|3,4", "b/k=5:6|c/u=7,8"],
            [
                ["a=1e-3,2e-3", "b=5e3,6e3"],
                ["a=1e-3,2e-3", "c=7e-6,8e-6"],
                ["a=3e-3,4e-3", "b=5e3,6e3"],
                ["a=3e-3,4e-3", "c=7e-6,8e-6"],
            ],
        ),
    ],
)
def test_expand_list(s: list[str], x: list[list[str]]):
    assert expand(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [  # pyright: ignore[reportUnknownArgumentType]
        (
            "a/m=1:2|3,4 b/k=5:6|c=7,8",
            [
                ["a=1e-3,2e-3", "b=5e3,6e3"],
                ["a=1e-3,2e-3", "c=7,8"],
                ["a=3e-3,4e-3", "b=5e3,6e3"],
                ["a=3e-3,4e-3", "c=7,8"],
            ],
        ),
        ("", [[]]),
    ],
)
def test_expand_str(s: str, x: list[list[str]]):
    assert expand(s) == x
