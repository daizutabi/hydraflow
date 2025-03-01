from typer.testing import CliRunner

from hydraflow.cli import app

runner = CliRunner()


def test_show():
    result = runner.invoke(app, ["show"])
    assert result.exit_code == 0
    assert "jobs:\n" in result.stdout
    assert "  args:\n" in result.stdout


def test_show_job():
    result = runner.invoke(app, ["show", "args"])
    assert result.exit_code == 0
    assert "name: args\n" in result.stdout
