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
    *,
    dry_run: Annotated[
        bool,
        Option("--dry-run", help="Perform a dry run"),
    ] = False,
) -> None:
    """Run a job."""
    import mlflow

    from hydraflow.executor.job import multirun, show

    job = get_job(name)
    if dry_run:
        show(job)
    else:
        mlflow.set_experiment(job.name)
        multirun(job)


@app.command()
def show(
    name: Annotated[str, Argument(help="Job name.", show_default=False)] = "",
) -> None:
    """Show the hydraflow config."""
    from omegaconf import OmegaConf

    if name:
        cfg = get_job(name)
    else:
        cfg = load_config()

    typer.echo(OmegaConf.to_yaml(cfg))


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
