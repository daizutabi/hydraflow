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


def test_none():
    cfg = load_config()
    assert cfg.jobs == {}


def test_job(config):
    cfg = config("jobs:\n  a:\n    run: a.test\n    add: --opt1 --opt2\n")
    assert cfg.jobs["a"].run == "a.test"
    assert cfg.jobs["a"].add == "--opt1 --opt2"


def test_set(config):
    cfg = config("jobs:\n  a:\n    sets:\n      - add: --opt1 --opt2\n")
    assert cfg.jobs["a"].sets[0].add == "--opt1 --opt2"
