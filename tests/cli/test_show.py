from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from hydraflow.cli import app

runner = CliRunner()


@pytest.mark.parametrize("file", ["hydraflow.yaml", "hydraflow.yml"])
def test_find_config(file):
    from hydraflow.cli import find_config

    Path(file).touch()
    assert find_config() == Path(file)


def test_find_config_error():
    from hydraflow.cli import find_config

    with pytest.raises(typer.Exit):
        find_config()


def test_load_config():
    from hydraflow.cli import load_config

    Path("hydraflow.yaml").write_text("a:\n b: 1")
    cfg = load_config()
    assert cfg["a"]["b"] == 1


def test_load_config_error():
    from hydraflow.cli import load_config

    Path("hydraflow.yml").write_text("- 1\n- 2")

    with pytest.raises(typer.Exit):
        load_config()


def test_invoke_error():
    result = runner.invoke(app, ["show"])
    assert result.exit_code == 1


def test_invoke():
    Path("hydraflow.yaml").write_text("a:\n b: [1, 2]")
    result = runner.invoke(app, ["show"])
    assert result.exit_code == 0
