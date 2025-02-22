import pytest
from omegaconf import DictConfig, OmegaConf

from hydraflow.jobs.conf import HydraflowConf
from hydraflow.jobs.io import load_config


@pytest.fixture(scope="module")
def schema():
    return OmegaConf.structured(HydraflowConf)


def test_scheme_type(schema: DictConfig):
    assert isinstance(schema, DictConfig)


def test_merge(schema: DictConfig):
    cfg = OmegaConf.merge(schema, {"run": "test"})
    assert cfg.run == "test"
    assert cfg.jobs == {}


def test_none():
    cfg = load_config()
    assert cfg.run == ""
    assert cfg.jobs == {}


def test_run(config):
    cfg = config("run: test\n")
    assert cfg.run == "test"
    assert cfg.jobs == {}


def test_job(config):
    cfg = config("jobs:\n  a:\n    run: a.test\n")
    assert cfg.run == ""
    assert cfg.jobs["a"].run == "a.test"


def test_step(config):
    cfg = config("jobs:\n  a:\n    steps:\n      - run: a.0.test\n")
    assert cfg.run == ""
    assert cfg.jobs["a"].run == ""
    assert cfg.jobs["a"].steps[0].run == "a.0.test"
