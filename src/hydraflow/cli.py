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
    names: Annotated[
        list[str] | None,
        Argument(help="Job names.", show_default=False),
    ] = None,
) -> None:
    """Run jobs."""
    typer.echo(names)

    cfg = load_config()
    typer.echo(cfg)


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
