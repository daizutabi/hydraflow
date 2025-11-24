from __future__ import annotations

import subprocess

from typer.testing import CliRunner

from hydraflow.cli import app

runner = CliRunner()


def test_invoke():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "hydraflow" in result.stdout
    assert result.stdout.count(".") == 2


def test_command():
    out = subprocess.check_output(["hydraflow", "--version"], text=True)
    assert "hydraflow" in out
    assert out.count(".") == 2
