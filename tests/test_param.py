from pathlib import Path

import mlflow
import pytest


@pytest.fixture
def param(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def param(value):
        monkeypatch.chdir(tmp_path)
        mlflow.set_experiment("test_param")

        with mlflow.start_run():
            mlflow.log_param("p", value, synchronous=True)

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
    from hydraflow.param import match

    p = param(x)
    assert p == y
    assert str(x) == y
    assert match(p, x)


def test_param_float():
    from hydraflow.param import match

    assert match("1.0", 1.0)
    assert match("1.0", 1)
    assert match("0.0", 0)
    assert match("0.0", 0.0)


def test_param_bool():
    from hydraflow.param import match

    assert not match("1", True)
    assert not match("0", False)


def test_match_list():
    from hydraflow.param import _match_list

    assert _match_list("1", [1, 2, 3]) is True
    assert _match_list("[1]", [1, 2, 3]) is None
    assert _match_list("(1,)", [1, 2, 3]) is None
    assert _match_list("{1: 3}", [1, 2, 3]) is None
    assert _match_list("2", [1, 2, 3]) is True
    assert _match_list("4", [1, 2, 3]) is False
    assert _match_list("4", [True]) is None
    assert _match_list("4", [None]) is None
    assert _match_list("4", ["4"]) is True
    assert _match_list("4", ["a"]) is False


def test_match_tuple():
    from hydraflow.param import _match_tuple

    assert _match_tuple("1", (1, 3)) is True
    assert _match_tuple("2", (1, 3)) is True
    assert _match_tuple("4", (1, 3)) is False
    assert _match_tuple("[1]", (1, 3)) is None
    assert _match_tuple("(1,)", (1, 3)) is None
    assert _match_tuple("{1: 3}", (1, 3)) is None
    assert _match_tuple("1", (True, False)) is None
    assert _match_tuple("1", (None, None)) is None
    assert _match_tuple("1", (1, 3.2)) is None


def test_to_value():
    from hydraflow.param import to_value

    assert to_value("1", int) == 1
    assert to_value("1", float) == 1.0
    assert to_value("1.0", float) == 1.0
    assert to_value("True", bool) is True
    assert to_value("False", bool) is False
    assert to_value("None", int) is None
    assert to_value("a", str) == "a"


def test_to_value_list():
    from hydraflow.param import to_value

    x = "[1, 2, 3]"
    assert to_value(x, list) == [1, 2, 3]
    x = "[1.2, 2.3, 3.4]"
    assert to_value(x, list) == [1.2, 2.3, 3.4]
    x = "[a, b, c]"
    assert to_value(x, list) == ["a", "b", "c"]
