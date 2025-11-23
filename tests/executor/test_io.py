from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from hydraflow.executor.conf import HydraflowConf
from hydraflow.executor.io import load_config


@pytest.fixture(scope="module")
def schema():
    return OmegaConf.structured(HydraflowConf)


def test_scheme_type(schema: DictConfig):
    assert isinstance(schema, DictConfig)


def test_merge(schema: DictConfig):
    cfg = OmegaConf.merge(schema, {})
    assert cfg.jobs == {}


def test_load_config_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    Path("hydraflow.yaml").write_text("- a\n- b\n")

    cfg = load_config()
    assert isinstance(cfg, DictConfig)
    assert cfg.jobs == {}


def test_load_config_none():
    cfg = load_config()
    assert cfg.jobs == {}


def test_load_config_job(tmp_path: Path):
    config_file = tmp_path.joinpath("hydraflow.yaml")
    config_file.write_text("jobs:\n  a:\n    run: a.test\n    add: --opt1 --opt2\n")
    cfg = load_config(config_file)
    assert cfg.jobs["a"].run == "a.test"
    assert cfg.jobs["a"].add == "--opt1 --opt2"


def test_load_config_set(tmp_path: Path):
    config_file = tmp_path.joinpath("hydraflow.yaml")
    config_file.write_text("jobs:\n  a:\n    sets:\n      - add: --opt1 --opt2\n")
    cfg = load_config(config_file)
    assert cfg.jobs["a"].sets[0].add == "--opt1 --opt2"


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
def test_load_config_args_quote(text: str, expected: str, tmp_path: Path):
    config_file = tmp_path.joinpath("hydraflow.yaml")
    config_file.write_text(f"jobs:\n  a:\n    sets:\n      - all: {text}\n")
    cfg = load_config(config_file)
    arg = cfg.jobs["a"].sets[0].all
    assert arg == expected
