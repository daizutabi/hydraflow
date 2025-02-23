import pytest


@pytest.mark.parametrize(("s", "x"), [("1", 1), ("1.2", 1.2), ("", 0)])
def test_to_number(s, x):
    from hydraflow.jobs.parser import to_number

    assert to_number(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [("1", 0), ("1.2", 1), ("1.234", 3), ("123.", 0), ("", 0)],
)
def test_num_point(s, x):
    from hydraflow.jobs.parser import num_point

    assert num_point(s) == x


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
    from hydraflow.jobs.parser import _get_range

    with pytest.raises(expected_exception) as excinfo:
        _get_range(arg)
    assert str(excinfo.value) == expected_message


@pytest.mark.parametrize(
    ("s", "x"),
    [
        ("1:2:3:suffix", ("1:2:3", "suffix")),
        ("1:2:3:s1", ("1:2:3", "s1")),
        ("1:2:3", ("1:2:3", "")),
        ("1.23", ("1.23", "")),
        ("1.23:s1", ("1.23", "s1")),
        ("1.23:s1:s2", ("1.23:s1", "s2")),
        ("1.23:s1:s2:s3", ("1.23:s1:s2", "s3")),
        ("1.23:s1:s2:s3:s4", ("1.23:s1:s2:s3", "s4")),
        ("1:2:3:M", ("1:2:3", "e6")),
        ("1:2:3:k", ("1:2:3", "e3")),
        ("1:2:3:m", ("1:2:3", "e-3")),
        ("1:2:3:n", ("1:2:3", "e-9")),
        ("1:2:3:p", ("1:2:3", "e-12")),
    ],
)
def test_split_suffix(s, x):
    from hydraflow.jobs.parser import split_suffix

    assert split_suffix(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [
        ("1", ["1"]),
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
        ("1:k", ["1e3"]),
        ("1:2:u", ["1e-6", "2e-6"]),
        ("1:.25:2:n", ["1e-9", "1.25e-9", "1.5e-9", "1.75e-9", "2.0e-9"]),
        ("1:2:e2", ["1e2", "2e2"]),
    ],
)
def test_collect_value(s, x):
    from hydraflow.jobs.parser import collect_values

    assert collect_values(s) == x


@pytest.mark.parametrize(
    ("s", "x"),
    [
        ("1,2,3", ["1", "2", "3"]),
        ("1:3,5:6", ["1", "2", "3", "5", "6"]),
        ("0:0.25:1,2.0", ["0", "0.25", "0.5", "0.75", "1.0", "2.0"]),
        ("3", ["3"]),
        ("3k", ["3e3"]),
        ("1:2:k,3:2:7:M", ["1e3", "2e3", "3e6", "5e6", "7e6"]),
        ("[1,2],[3,4]", ["[1,2]", "[3,4]"]),
        ("'1,2','3,4'", ["'1,2'", "'3,4'"]),
        ('"1,2","3,4"', ['"1,2"', '"3,4"']),
    ],
)
def test_expand_value(s, x):
    from hydraflow.jobs.parser import expand_values, parse

    assert expand_values(s) == x
    assert parse(s) == ",".join(x)
