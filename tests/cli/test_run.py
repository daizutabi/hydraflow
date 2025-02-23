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
