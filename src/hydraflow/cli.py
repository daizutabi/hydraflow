"""Hydraflow CLI."""

from __future__ import annotations

import shlex
from typing import Annotated

import typer
from typer import Argument, Exit, Option

from hydraflow.executor.conf import Job

app = typer.Typer(add_completion=False)


@app.command(context_settings={"ignore_unknown_options": True})
def run(
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
    from hydraflow.executor.job import iter_batches, iter_calls, iter_runs

    args = args or []
    job = get_job(name)

    if job.submit:
        submit(job, args, dry_run=dry_run)
        raise Exit

    if job.run:
        args = [*shlex.split(job.run), *args]
        it = iter_runs(args, iter_batches(job), dry_run=dry_run)
    elif job.call:
        args = [*shlex.split(job.call), *args]
        it = iter_calls(args, iter_batches(job), dry_run=dry_run)
    else:
        typer.echo(f"No command found in job: {job.name}.")
        raise Exit(1)

    if not dry_run:
        import mlflow

        mlflow.set_experiment(job.name)

    for task in it:  # jobs will be executed here
        if job.run and dry_run:
            typer.echo(shlex.join(task.args))
        elif job.call and dry_run:
            funcname, *args = task.args
            arg = ", ".join(f"{arg!r}" for arg in args)
            typer.echo(f"{funcname}([{arg}])")


def submit(job: Job, args: list[str], *, dry_run: bool) -> None:
    """Submit a job."""
    from hydraflow.executor.job import iter_batches, submit

    if not dry_run:
        import mlflow

        mlflow.set_experiment(job.name)

    args = [*shlex.split(job.submit), *args]
    result = submit(args, iter_batches(job), dry_run=dry_run)

    if dry_run and isinstance(result, tuple):
        for line in result[1].splitlines():
            args = shlex.split(line)
            typer.echo(shlex.join([*result[0][:-1], *args]))


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
        raise Exit
