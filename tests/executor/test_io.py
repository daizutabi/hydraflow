from pathlib import Path

import pytest
from omegaconf import DictConfig


@pytest.mark.parametrize("file", ["hydraflow.yaml", "hydraflow.yml"])
def test_find_config(file, chdir):
    from hydraflow.executor.io import find_config_file

    Path(file).touch()
    assert find_config_file() == Path(file)
    Path(file).unlink()


def test_find_config_none(chdir):
    from hydraflow.executor.io import find_config_file

    assert find_config_file() is None


def test_load_config_list(chdir):
    from hydraflow.executor.io import load_config

    Path("hydraflow.yaml").write_text("- a\n- b\n")

    cfg = load_config()
    assert isinstance(cfg, DictConfig)
    assert cfg.jobs == {}

    Path("hydraflow.yaml").unlink()
