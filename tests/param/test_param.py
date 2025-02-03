from pathlib import Path

import mlflow
import pytest

from hydraflow.param import match

pytestmark = pytest.mark.xdist_group(name="group2")


@pytest.fixture
def param(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def param(value):
        monkeypatch.chdir(tmp_path)
        mlflow.set_experiment("test_param")

        with mlflow.start_run():
            mlflow.log_param("p", value)

        runs = mlflow.search_runs(output_format="list")
        p = runs[0].data.params["p"]
        assert isinstance(p, str)
        return p

    return param


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (1, "1"),
        (1.0, "1.0"),
        ("1", "1"),
        ("a", "a"),
        ("'a'", "'a'"),
        ('"a"', '"a"'),
        (True, "True"),
        (False, "False"),
        (None, "None"),
        ([], "[]"),
        ((), "()"),
        ({}, "{}"),
        ([1, 2, 3], "[1, 2, 3]"),
        (["1", "2", "3"], "['1', '2', '3']"),
        (("1", "2", "3"), "('1', '2', '3')"),
        ({"a": 1, "b": "c"}, "{'a': 1, 'b': 'c'}"),
    ],
)
def test_param(param, x, y):
    p = param(x)
    assert p == y
    assert str(x) == y
    assert match(p, x)


@pytest.mark.parametrize(
    ("param", "value"),
    [("1.0", lambda x: float(x) > 0), ("-1.0", lambda x: float(x) < 0)],
)
def test_match_callable(param, value):
    assert match(param, value)


@pytest.mark.parametrize(
    ("param", "value"),
    [("1.0", 1.0), ("1.0", 1), ("0.0", 0), ("0.0", 0.0)],
)
def test_match_float(param, value):
    assert match(param, value)


@pytest.mark.parametrize(
    ("param", "value"),
    [("1", True), ("0", False)],
)
def test_match_bool(param, value):
    assert not match(param, value)


@pytest.mark.parametrize(
    ("param", "value", "result"),
    [
        ("1", [1, 2, 3], True),
        ("[1]", [1, 2, 3], None),
        ("(1,)", [1, 2, 3], None),
        ("{1: 3}", [1, 2, 3], None),
        ("2", [1, 2, 3], True),
        ("4", [1, 2, 3], False),
        ("4", [True], None),
        ("4", [None], None),
        ("4", ["4"], True),
        ("4", ["a"], False),
    ],
)
def test_match_list(param, value, result):
    from hydraflow.param import _match_list, match

    assert _match_list(param, value) is result
    if result is not None:
        assert match(param, value) is result


@pytest.mark.parametrize(
    ("param", "value", "result"),
    [
        ("1", (1, 3), True),
        ("2", (1, 3), True),
        ("4", (1, 3), False),
        ("[1]", (1, 3), None),
        ("(1,)", (1, 3), None),
        ("{1: 3}", (1, 3), None),
        ("1", (True, False), None),
        ("1", (None, None), None),
        ("1", (1, 3.2), None),
    ],
)
def test_match_tuple(param, value, result):
    from hydraflow.param import _match_tuple, match

    assert _match_tuple(param, value) is result
    if result is not None:
        assert match(param, value) is result


@pytest.mark.parametrize(
    ("param", "type_", "result"),
    [("1", int, 1), ("1", float, 1.0), ("1.0", float, 1.0), ("a", str, "a")],
)
def test_to_value_eq(param, type_, result):
    from hydraflow.param import to_value

    v = to_value(param, type_)
    assert v == result
    assert str(v) == str(result)


@pytest.mark.parametrize(
    ("param", "type_", "result"),
    [("True", bool, True), ("False", bool, False), ("None", int, None)],
)
def test_to_value_is(param, type_, result):
    from hydraflow.param import to_value

    assert to_value(param, type_) is result


@pytest.mark.parametrize(
    ("param", "result"),
    [
        ("[1, 2, 3]", [1, 2, 3]),
        ("[1.2, 2.3, 3.4]", [1.2, 2.3, 3.4]),
        ("[a, b, c]", ["a", "b", "c"]),
    ],
)
def test_to_value_list(param, result):
    from hydraflow.param import to_value

    assert to_value(param, list) == result
