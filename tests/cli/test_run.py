from __future__ import annotations

from typing import Any

from typer.testing import CliRunner

from hydraflow.cli import app
from hydraflow.core.io import iter_run_dirs
from hydraflow.core.run import Run

runner = CliRunner()


def test_run_args_dry_run() -> None:
    result = runner.invoke(app, ["run", "args", "--dry-run"])
    assert result.exit_code == 0
    out = result.stdout
    assert "app.py --multirun count=1,2,3 name=a,b" in out
    assert "app.py --multirun count=4,5,6 name=c,d" in out
    assert out.count("hydra.job.name=args") == 2


def test_run_batch_dry_run() -> None:
    result = runner.invoke(app, ["run", "batch", "--dry-run"])
    assert result.exit_code == 0
    out = result.stdout
    assert "name=a count=1,2" in out
    assert "name=b count=1,2" in out
    assert "name=c,d count=100" in out
    assert "name=e,f count=100" in out
    assert out.count("hydra.job.name=batch") == 4


def test_run_parallel_dry_run() -> None:
    result = runner.invoke(app, ["run", "parallel", "--dry-run"])
    assert result.exit_code == 0
    out = result.stdout
    lines = out.splitlines()
    assert len(lines) == 2
    assert "count=1,2,3,4" in lines[0]
    assert "hydra.launcher.n_jobs=2" in lines[0]
    assert "count=11,12,13,14" in lines[1]
    assert "hydra.launcher.n_jobs=4" in lines[1]


def test_run_parallel_dry_run_extra_args() -> None:
    args = ["run", "parallel", "--dry-run", "a", "--b", "--", "--dry-run"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert result.stdout.count("app.py a --b --dry-run --multirun") == 2


def test_run_echo_dry_run() -> None:
    args = ["run", "echo", "--dry-run"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert result.stdout.count("typer.echo(['a', 'b', 'c', '--multirun',") == 4


def test_submit_dry_run() -> None:
    args = ["run", "submit", "--dry-run", "a", "--b", "--", "--dry-run"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert result.stdout.count("submit.py a --b --dry-run") == 1
    assert result.stdout.count("--multirun") == 4
    lines = result.stdout.splitlines()
    assert len(lines) == 5
    assert "name=a count=1" in lines[1]
    assert "name=b count=1" in lines[2]
    assert "name=c count=5" in lines[3]
    assert "name=d count=6" in lines[4]


def test_run_args() -> None:
    result = runner.invoke(app, ["run", "args"])
    assert result.exit_code == 0
    run_dirs = list(iter_run_dirs("args"))
    assert len(run_dirs) == 12


def test_run_batch() -> None:
    result = runner.invoke(app, ["run", "batch"])
    assert result.exit_code == 0
    run_dirs = list(iter_run_dirs("batch"))
    assert len(run_dirs) == 8


def test_run_parallel() -> None:
    result = runner.invoke(app, ["run", "parallel"])
    assert result.exit_code == 0
    run_dirs = list(iter_run_dirs("parallel"))
    assert len(run_dirs) == 8

    result = runner.invoke(app, ["run", "parallel"])  # skip if already run
    assert result.exit_code == 0
    run_dirs = list(iter_run_dirs("parallel"))
    assert len(run_dirs) == 8


def test_run_echo() -> None:
    result = runner.invoke(app, ["run", "echo"])
    assert result.exit_code == 0
    out = result.stdout
    lines = out.splitlines()
    assert "['a', 'b', 'c', '--multirun', 'name=a', 'count=1,2,3'" in lines[-4]
    assert "['a', 'b', 'c', '--multirun', 'name=b', 'count=1,2,3'" in lines[-3]
    assert "['a', 'b', 'c', '--multirun', 'name=c', 'count=4,5,6'" in lines[-2]
    assert "['a', 'b', 'c', '--multirun', 'name=d', 'count=4,5,6'" in lines[-1]


def test_submit() -> None:
    result = runner.invoke(app, ["run", "submit"])
    assert result.exit_code == 0
    run_dirs = list(iter_run_dirs("submit"))
    assert len(run_dirs) == 4

    result = runner.invoke(app, ["run", "submit"])  # skip if already run
    assert result.exit_code == 0
    run_dirs = list(iter_run_dirs("submit"))
    assert len(run_dirs) == 4


def test_run() -> None:
    runner.invoke(app, ["run", "job-name"])
    run_dir = next(iter_run_dirs("job-name"))
    run: Run[Any, None] = Run(run_dir)
    assert run.info.job_name == "job-name"


def test_run_error() -> None:
    result = runner.invoke(app, ["run", "error"])
    assert result.exit_code == 1
    assert "No command found in job: error" in result.stdout


def test_run_error_not_found() -> None:
    result = runner.invoke(app, ["run", "not_found"])
    assert result.exit_code == 1
    assert "Job not found: not_found" in result.stdout
