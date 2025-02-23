import pytest


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a=1", "a=1"),
        ("'a=1'", "a=1"),
        ('"a=1"', "a=1"),
        ("'\"a=1\"'", '"a=1"'),
        ("\"'a=1'\"", "'a=1'"),
        ("a='1,2'", "a='1,2'"),
        ("a|b", "a|b"),
        ("a:b", "a:b"),
        ("a[b]", "a[b]"),
    ],
)
def test_args_quote(args, text, expected):
    assert args(text) == expected
