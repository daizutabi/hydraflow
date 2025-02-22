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
    cfg: HydraflowConf = OmegaConf.merge(schema, {"run": "test"})  # type: ignore
    assert cfg.run == "test"
    assert cfg.jobs == {}


def test_none():
    cfg = load_config()
    assert cfg.run == ""
    assert cfg.jobs == {}


def test_run(save):
    save("run: test\n")

    cfg = load_config()
    assert cfg.run == "test"
    assert cfg.jobs == {}


def test_job(save):
    save("jobs:\n  a:\n    run: a.test\n")

    cfg = load_config()
    assert cfg.run == ""
    assert cfg.jobs["a"].run == "a.test"


def test_step(save):
    save("jobs:\n  a:\n    steps:\n      - run: a.0.test\n")

    cfg = load_config()
    assert cfg.run == ""
    assert cfg.jobs["a"].run == ""
    assert cfg.jobs["a"].steps[0].run == "a.0.test"
