"""Hydraflow CLI."""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Annotated

import typer
from omegaconf.errors import ConfigKeyError
from typer import Argument, Exit, Option

if TYPE_CHECKING:
    from hydraflow.executor.conf import Job

app = typer.Typer(add_completion=False)


@app.command("run", context_settings={"ignore_unknown_options": True})
def _run(  # pyright: ignore[reportUnusedFunction]
    name: Annotated[str, Argument(help="Job name.", show_default=False)],
    *,
    args: Annotated[
        list[str] | None,
        Argument(help="Arguments to pass to the job.", show_default=False),
    ] = None,
    dry_run: Annotated[
        bool,
        Option("--dry-run", help="Perform a dry run."),
    ] = False,
) -> None:
    """Run a job."""
    from hydraflow.executor.io import get_job

    args = args or []
    try:
        job = get_job(name)
    except ConfigKeyError:
        typer.echo(f"Job not found: {name}")
        raise typer.Exit(1) from None

    if job.submit:
        submit(job, args, dry_run=dry_run)

    elif job.run:
        run(job, args, dry_run=dry_run)

    elif job.call:
        call(job, args, dry_run=dry_run)

    else:
        typer.echo(f"No command found in job: {job.name}")
        raise Exit(1)


def run(job: Job, args: list[str], *, dry_run: bool) -> None:
    """Run a job."""
    from hydraflow.executor import aio
    from hydraflow.executor.job import iter_batches, iter_tasks

    args = [*shlex.split(job.run), *args]
    it = iter_tasks(args, iter_batches(job))

    if not dry_run:
        aio.run(it)
        raise Exit

    for task in it:
        typer.echo(shlex.join(task.args))


def call(job: Job, args: list[str], *, dry_run: bool) -> None:
    """Call a job."""
    from hydraflow.executor.job import iter_batches, iter_calls

    args = [*shlex.split(job.call), *args]
    it = iter_calls(args, iter_batches(job))

    if not dry_run:
        for call in it:
            call.func()
        raise Exit

    for task in it:
        funcname, *args = task.args
        arg = ", ".join(f"{arg!r}" for arg in args)
        typer.echo(f"{funcname}([{arg}])")


def submit(job: Job, args: list[str], *, dry_run: bool) -> None:
    """Submit a job."""
    from hydraflow.executor.job import iter_batches, submit

    args = [*shlex.split(job.submit), *args]
    result = submit(args, iter_batches(job), dry_run=dry_run)

    if dry_run and isinstance(result, tuple):
        typer.echo(shlex.join(result[0]))
        typer.echo(result[1])


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
    """Hydraflow CLI."""
    if version:
        import importlib.metadata

        typer.echo(f"hydraflow {importlib.metadata.version('hydraflow')}")
        raise Exit
