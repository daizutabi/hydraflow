from pathlib import Path


def test_equals_config():
    from hydraflow.core.main import equals

    assert equals(Path("a/b/c"), {"a": 1}, None) is False


def test_equals_overrides():
    from hydraflow.core.main import equals

    assert equals(Path("a/b/c"), {"a": 1}, ["a=1"]) is False
