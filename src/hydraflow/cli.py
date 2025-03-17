"""Hydraflow CLI."""

from __future__ import annotations

import shlex
from typing import Annotated

import typer
from rich.console import Console
from typer import Argument, Option

app = typer.Typer(add_completion=False)
console = Console()


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

    from hydraflow.executor.io import get_job
    from hydraflow.executor.job import iter_batches, multicall, multirun, to_text

    job = get_job(name)

    if dry_run:
        typer.echo(to_text(job))
        raise typer.Exit

    import mlflow

    mlflow.set_experiment(job.name)

    if job.run:
        executable, *args = shlex.split(job.run)
        it = multirun(executable, args, iter_batches(job))
    elif job.call:
        funcname, *args = shlex.split(job.call)
        it = multicall(funcname, args, iter_batches(job))
    else:
        typer.echo(f"No run or call found in job: {job.name}.")
        raise typer.Exit(1)

    for _ in it:
        pass


@app.command()
def show(
    name: Annotated[str, Argument(help="Job name.", show_default=False)] = "",
) -> None:
    """Show the hydraflow config."""
    from omegaconf import OmegaConf

    from hydraflow.executor.io import get_job, load_config

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
