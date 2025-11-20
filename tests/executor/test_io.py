from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig

from hydraflow.executor.io import find_config_file, load_config


@pytest.mark.parametrize("file", ["hydraflow.yaml", "hydraflow.yml"])
def test_find_config(file: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    Path(file).touch()
    assert find_config_file() == Path(file)
    Path(file).unlink()


def test_find_config_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    assert find_config_file() is None


def test_load_config_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    Path("hydraflow.yaml").write_text("- a\n- b\n")

    cfg = load_config()
    assert isinstance(cfg, DictConfig)
    assert cfg.jobs == {}

    Path("hydraflow.yaml").unlink()
