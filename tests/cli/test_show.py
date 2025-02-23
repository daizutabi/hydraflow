from typer.testing import CliRunner

from hydraflow.cli import app

runner = CliRunner()


def test_show_args():
    result = runner.invoke(app, ["show", "args"])
    assert result.exit_code == 0
    assert "hydra.job.name=args" in result.stdout
    assert "count=1,2,3 name=a,b" in result.stdout
    assert "count=4,5,6 name=c,d" in result.stdout


def test_show_batch():
    result = runner.invoke(app, ["show", "batch"])
    assert result.exit_code == 0
    assert "hydra.job.name=batch" in result.stdout
    assert "count=1,2 name=a" in result.stdout
    assert "count=1,2 name=b" in result.stdout
    assert "count=100 name=c,d" in result.stdout
    assert "count=100 name=e,f" in result.stdout
