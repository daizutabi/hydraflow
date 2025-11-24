from __future__ import annotations

from pathlib import Path

from hydraflow.core.main import equals, get_run_id


def test_get_run_id():
    assert get_run_id("invalid", None, None) is None


def test_equals_config():
    assert equals(Path("a/b/c"), {"a": 1}, None) is False


def test_equals_overrides():
    assert equals(Path("a/b/c"), {"a": 1}, ["a=1"]) is False
