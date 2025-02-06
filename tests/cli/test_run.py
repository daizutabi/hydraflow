from pathlib import Path

from typer.testing import CliRunner

from hydraflow.cli import app

runner = CliRunner()


def test_invoke_error():
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 1


def test_invoke():
    Path("hydraflow.yaml").write_text("a:\n b: [1, 2]")
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 0
