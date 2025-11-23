from __future__ import annotations

from pathlib import Path

from hydraflow.core.main import equals


def test_equals_config():
    assert equals(Path("a/b/c"), {"a": 1}, None) is False


def test_equals_overrides():
    assert equals(Path("a/b/c"), {"a": 1}, ["a=1"]) is False
