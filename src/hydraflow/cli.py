"""Hydraflow CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from typer import Argument, Option

from hydraflow.executor.io import load_config

if TYPE_CHECKING:
    from hydraflow.executor.job import Job

app = typer.Typer(add_completion=False)
console = Console()


def get_job(name: str) -> Job:
    cfg = load_config()
    job = cfg.jobs[name]

    if not job.name:
        job.name = name

    return job


@app.command()
def run(
    name: Annotated[str, Argument(help="Job name.", show_default=False)],
) -> None:
    """Run a job."""
    import mlflow

    from hydraflow.executor.job import multirun

    job = get_job(name)
    mlflow.set_experiment(job.name)
    multirun(job)


@app.command()
def show(
    name: Annotated[str, Argument(help="Job name.", show_default=False)],
) -> None:
    """Show a job."""
    from hydraflow.executor.job import show

    job = get_job(name)
    show(job)


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
