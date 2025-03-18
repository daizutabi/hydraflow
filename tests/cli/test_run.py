import pytest
from typer.testing import CliRunner

import hydraflow
from hydraflow.cli import app

runner = CliRunner()


def test_run_args_dry_run():
    result = runner.invoke(app, ["run", "args", "--dry-run"])
    assert result.exit_code == 0
    out = result.stdout
    assert "app.py --multirun count=1,2,3 name=a,b" in out
    assert "app.py --multirun count=4,5,6 name=c,d" in out
    assert out.count("hydra.job.name=args") == 2


def test_run_batch_dry_run():
    result = runner.invoke(app, ["run", "batch", "--dry-run"])
    assert result.exit_code == 0
    out = result.stdout
    assert "name=a count=1,2" in out
    assert "name=b count=1,2" in out
    assert "name=c,d count=100" in out
    assert "name=e,f count=100" in out
    assert out.count("hydra.job.name=batch") == 4


def test_run_parallel_dry_run():
    result = runner.invoke(app, ["run", "parallel", "--dry-run"])
    assert result.exit_code == 0
    out = result.stdout
    lines = out.splitlines()
    assert len(lines) == 2
    assert "count=1,2,3,4" in lines[0]
    assert "hydra.launcher.n_jobs=2" in lines[0]
    assert "count=11,12,13,14" in lines[1]
    assert "hydra.launcher.n_jobs=4" in lines[1]


def test_run_parallel_dry_run_extra_args():
    args = ["run", "parallel", "--dry-run", "a", "--b", "--", "--dry-run"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert result.stdout.count("app.py a --b --dry-run --multirun") == 2


def test_run_echo_dry_run():
    args = ["run", "echo", "--dry-run"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert result.stdout.count("typer.echo(['a', 'b', 'c', '--multirun',") == 4


def test_submit_dry_run():
    args = ["run", "submit", "--dry-run", "a", "--b", "--", "--dry-run"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert result.stdout.count("submit.py a --b --dry-run --multirun") == 4


@pytest.mark.xdist_group(name="group1")
def test_run_args():
    result = runner.invoke(app, ["run", "args"])
    assert result.exit_code == 0
    run_ids = hydraflow.list_run_ids("args")
    assert len(run_ids) == 12


@pytest.mark.xdist_group(name="group2")
def test_run_batch():
    result = runner.invoke(app, ["run", "batch"])
    assert result.exit_code == 0
    run_ids = hydraflow.list_run_ids("batch")
    assert len(run_ids) == 8


@pytest.mark.xdist_group(name="group3")
def test_run_parallel():
    result = runner.invoke(app, ["run", "parallel"])
    assert result.exit_code == 0
    run_ids = hydraflow.list_run_ids("parallel")
    assert len(run_ids) == 8

    result = runner.invoke(app, ["run", "parallel"])  # skip if already run
    assert result.exit_code == 0
    run_ids = hydraflow.list_run_ids("parallel")
    assert len(run_ids) == 8


@pytest.mark.xdist_group(name="group4")
def test_run_echo():
    result = runner.invoke(app, ["run", "echo"])
    assert result.exit_code == 0
    out = result.stdout
    lines = out.splitlines()
    assert len(lines) == 5
    assert "['a', 'b', 'c', '--multirun', 'name=a', 'count=1,2,3'" in lines[1]
    assert "['a', 'b', 'c', '--multirun', 'name=b', 'count=1,2,3'" in lines[2]
    assert "['a', 'b', 'c', '--multirun', 'name=c', 'count=4,5,6'" in lines[3]
    assert "['a', 'b', 'c', '--multirun', 'name=d', 'count=4,5,6'" in lines[4]


@pytest.mark.xdist_group(name="group4")
def test_submit():
    result = runner.invoke(app, ["run", "submit"])
    assert result.exit_code == 0
    out = result.stdout
    lines = out.splitlines()
    assert len(lines) == 1
    run_ids = hydraflow.list_run_ids("submit")
    assert len(run_ids) == 4

    result = runner.invoke(app, ["run", "submit"])  # skip if already run
    assert result.exit_code == 0
    run_ids = hydraflow.list_run_ids("submit")
    assert len(run_ids) == 4


@pytest.mark.xdist_group(name="group5")
def test_run_error():
    result = runner.invoke(app, ["run", "error"])
    assert result.exit_code == 1
    assert "No command found in job: error." in result.stdout
