from typer.testing import CliRunner

from hydraflow.cli import app

runner = CliRunner()


def test_show():
    result = runner.invoke(app, ["show"])
    assert result.exit_code == 0
    assert "jobs:\n" in result.stdout
    assert "  args:\n" in result.stdout


def test_run_show_args():
    result = runner.invoke(app, ["run", "args", "--show"])
    assert result.exit_code == 0
    assert "hydra.job.name=args" in result.stdout
    assert "count=1,2,3 name=a,b" in result.stdout
    assert "count=4,5,6 name=c,d" in result.stdout


def test_run_show_batch():
    result = runner.invoke(app, ["run", "batch", "--show"])
    assert result.exit_code == 0
    assert "hydra.job.name=batch" in result.stdout
    assert "name=a count=1,2" in result.stdout
    assert "name=b count=1,2" in result.stdout
    assert "count=10 name=c,d" in result.stdout
    assert "count=11 name=c,d" in result.stdout
