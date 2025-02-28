import pytest


@pytest.mark.parametrize(("s", "x"), [("1", 1), ("1.2", 1.2), ("", 0)])
def test_to_number(s, x):
    from hydraflow.executor.parser import to_number

    assert to_number(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [("1", 0), ("1.2", 1), ("1.234", 3), ("123.", 0), ("", 0), ("1.234e-10", 3)],
)
def test_count_decimal_places(s, x):
    from hydraflow.executor.parser import count_decimal_places

    assert count_decimal_places(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [
        ("1:2", (1, 1, 2)),
        (":2", (0, 1, 2)),
        ("0.1:0.1:0.4", (0.1, 0.1, 0.4)),
    ],
)
def test_get_range(s, x):
    from hydraflow.executor.parser import _get_range

    assert _get_range(s) == x


@pytest.mark.parametrize(
    ("arg", "expected_exception", "expected_message"),
    [
        ("5:3", ValueError, "start cannot be greater than stop"),
        ("1:0:2", ValueError, "step cannot be zero"),
        ("3:-1:5", ValueError, "start cannot be less than stop"),
        ("4.5:1.0:3.5", ValueError, "start cannot be greater than stop"),
    ],
)
def test_get_range_errors(arg, expected_exception, expected_message):
    from hydraflow.executor.parser import _get_range

    with pytest.raises(expected_exception) as excinfo:
        _get_range(arg)
    assert str(excinfo.value) == expected_message


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
def test_split_suffix(s, x):
    from hydraflow.executor.parser import split_suffix

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
        ("0:0.25:1", ["0", "0.25", "0.5", "0.75", "1.0"]),
        (":3", ["0", "1", "2", "3"]),
        ("5:7", ["5", "6", "7"]),
        ("-1:1", ["-1", "0", "1"]),
        ("1:0.5:2", ["1", "1.5", "2.0"]),
        ("1.:0.5:2", ["1.0", "1.5", "2.0"]),
        ("2:0.5:3", ["2", "2.5", "3.0"]),
        ("-1:0.5:1", ["-1", "-0.5", "0.0", "0.5", "1.0"]),
        ("4:-1:2", ["4", "3", "2"]),
        ("4.5:-1.5:2", ["4.5", "3.0"]),
        ("4.5:-1.5:1.5", ["4.5", "3.0", "1.5"]),
        ("4.5:-1.5:-4.5", ["4.5", "3.0", "1.5", "0.0", "-1.5", "-3.0", "-4.5"]),
        ("1:2:u", ["1e-6", "2e-6"]),
        ("1:.25:2:n", ["1e-9", "1.25e-9", "1.5e-9", "1.75e-9", "2.0e-9"]),
        ("1:2:e2", ["1e2", "2e2"]),
        (":2:e2", ["0", "1e2", "2e2"]),
        ("-2:2:k", ["-2e3", "-1e3", "0", "1e3", "2e3"]),
    ],
)
def test_collect_value(s, x):
    from hydraflow.executor.parser import collect_values

    assert collect_values(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [
        ("1,2,3", ["1", "2", "3"]),
        ("1:3,5:6", ["1", "2", "3", "5", "6"]),
        ("0:0.25:1,2.0", ["0", "0.25", "0.5", "0.75", "1.0", "2.0"]),
        ("3", ["3"]),
        ("3:k", ["3e3"]),
        ("1:3:k,3:2:7:M", ["1e3", "2e3", "3e3", "3e6", "5e6", "7e6"]),
        ("1:M,3:2:7:M", ["1e6", "3e6", "5e6", "7e6"]),
        ("[1,2],[3,4]", ["[1,2]", "[3,4]"]),
        ("'1,2','3,4'", ["'1,2'", "'3,4'"]),
        ('"1,2","3,4"', ['"1,2"', '"3,4"']),
    ],
)
def test_expand_value(s, x):
    from hydraflow.executor.parser import expand_values

    assert expand_values(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [
        ("a=1", "a=1"),
        ("a=1,2", "a=1,2"),
        ("a=1:2", "a=1,2"),
        ("a=:2:3", "a=0,2"),
        ("a=1:3:k", "a=1e3,2e3,3e3"),
        ("a=1:3:k,2:4:M", "a=1e3,2e3,3e3,2e6,3e6,4e6"),
    ],
)
def test_collect_arg(s, x):
    from hydraflow.executor.parser import collect_arg

    assert collect_arg(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [
        ("a=1", ["a=1"]),
        ("a=1,2", ["a=1", "a=2"]),
        ("a=1:2", ["a=1", "a=2"]),
        ("a=:2:3", ["a=0", "a=2"]),
        ("a=1:3:k", ["a=1e3", "a=2e3", "a=3e3"]),
        ("a=1:3:k,2:4:M", ["a=1e3", "a=2e3", "a=3e3", "a=2e6", "a=3e6", "a=4e6"]),
        ("a=1,2|3,4", ["a=1,2", "a=3,4"]),
        ("a=1:4|3:5:m", ["a=1,2,3,4", "a=3e-3,4e-3,5e-3"]),
        ("a=1,2|b=3,4|c=5,6", ["a=1,2", "b=3,4", "c=5,6"]),
    ],
)
def test_expand_arg(s, x):
    from hydraflow.executor.parser import expand_arg

    assert list(expand_arg(s)) == x


def test_expand_arg_error():
    from hydraflow.executor.parser import expand_arg

    with pytest.raises(ValueError):
        list(expand_arg("1,2|3,4|"))


@pytest.mark.parametrize(
    ("s", "x"),
    [
        (["a=1", "b"], ["a=1"]),
        (["a=1:3"], ["a=1,2,3"]),
        (["a=1:3", "b=4:6"], ["a=1,2,3", "b=4,5,6"]),
        ("a=1:3\nb=4:6", ["a=1,2,3", "b=4,5,6"]),
        ("", []),
    ],
)
def test_collect(s, x):
    from hydraflow.executor.parser import collect

    assert collect(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [
        (["a=1", "b"], [["a=1"]]),
        (["a=1,2"], [["a=1"], ["a=2"]]),
        (
            " a=1,2\n b=3,4\n",
            [["a=1", "b=3"], ["a=1", "b=4"], ["a=2", "b=3"], ["a=2", "b=4"]],
        ),
        (["a=1:2|3,4"], [["a=1,2"], ["a=3,4"]]),
        (
            ["a=1:2|3,4", "b=5:6|c=7,8"],
            [
                ["a=1,2", "b=5,6"],
                ["a=1,2", "c=7,8"],
                ["a=3,4", "b=5,6"],
                ["a=3,4", "c=7,8"],
            ],
        ),
        ("", [[]]),
    ],
)
def test_expand(s, x):
    from hydraflow.executor.parser import expand

    assert expand(s) == x
