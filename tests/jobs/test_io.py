from pathlib import Path

import pytest


@pytest.mark.parametrize("file", ["hydraflow.yaml", "hydraflow.yml"])
def test_find_config(file, chdir):
    from hydraflow.jobs.io import find_config_file

    Path(file).touch()
    assert find_config_file() == Path(file)
    Path(file).unlink()


def test_find_config_none(chdir):
    from hydraflow.jobs.io import find_config_file

    assert find_config_file() is None
