"""Hydraflow CLI."""

from __future__ import annotations

from typing import Annotated

import typer
from omegaconf import OmegaConf
from rich.console import Console
from typer import Argument, Option

from hydraflow.executor.io import load_config

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def run(
    name: Annotated[str, Argument(help="Job name.", show_default=False)],
    show_: Annotated[
        bool,
        Option("--show", help="Show the job and exit."),
    ] = False,
) -> None:
    """Run a job."""
    from hydraflow.executor.job import multirun, show

    cfg = load_config()
    job = cfg.jobs[name]
    if not job.name:
        job.name = name

    if show_:
        show(job)
        raise typer.Exit

    multirun(job)


@app.command()
def show() -> None:
    """Show the config."""
    cfg = load_config()
    code = OmegaConf.to_yaml(cfg)
    typer.echo(code)


@app.callback(invoke_without_command=True)
def callback(
    *,
    version: Annotated[
        bool,
        Option("--version", help="Show the version and exit."),
    ] = False,
) -> None:
    if version:
        import importlib.metadata

        typer.echo(f"hydraflow {importlib.metadata.version('hydraflow')}")
        raise typer.Exit
