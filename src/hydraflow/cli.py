"""Hydraflow CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from typer import Argument, Option

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
    from rich.syntax import Syntax

    cfg = load_config()
    code = OmegaConf.to_yaml(cfg)
    syntax = Syntax(code, "yaml")
    console.print(syntax)


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


def find_config() -> Path:
    if Path("hydraflow.yaml").exists():
        return Path("hydraflow.yaml")

    if Path("hydraflow.yml").exists():
        return Path("hydraflow.yml")

    typer.echo("No config file found.")
    raise typer.Exit(code=1)


def load_config() -> DictConfig:
    cfg = OmegaConf.load(find_config())

    if isinstance(cfg, DictConfig):
        return cfg

    typer.echo("Invalid config file.")
    raise typer.Exit(code=1)
