import pytest
from typer.testing import CliRunner

import hydraflow
from hydraflow.cli import app

pytestmark = pytest.mark.xdist_group(name="group1")

runner = CliRunner()


def test_run_args():
    result = runner.invoke(app, ["run", "args"])
    assert result.exit_code == 0
    run_ids = hydraflow.list_run_ids("args")
    assert len(run_ids) == 12


def test_run_batch():
    result = runner.invoke(app, ["run", "batch"])
    assert result.exit_code == 0
    run_ids = hydraflow.list_run_ids("batch")
    assert len(run_ids) == 8


def test_run_args_dry_run():
    result = runner.invoke(app, ["run", "args", "--dry-run"])
    assert result.exit_code == 0
    assert "hydra.job.name=args" in result.stdout
    assert "count=1,2,3 name=a,b" in result.stdout
    assert "count=4,5,6 name=c,d" in result.stdout


def test_run_batch_dry_run():
    result = runner.invoke(app, ["run", "batch", "--dry-run"])
    assert result.exit_code == 0
    assert "hydra.job.name=batch" in result.stdout
    assert "count=1,2 name=a" in result.stdout
    assert "count=1,2 name=b" in result.stdout
    assert "count=100 name=c,d" in result.stdout
    assert "count=100 name=e,f" in result.stdout
