"""Job execution and argument handling for HydraFlow.

This module provides functionality for executing jobs in HydraFlow, including:

- Argument parsing and expansion for job steps
- Batch processing of Hydra configurations
- Execution of jobs via shell commands or Python functions

The module supports two execution modes:

1. Shell command execution
2. Python function calls

Each job can consist of multiple steps, and each step can have its own
arguments and configurations that will be expanded into multiple runs.
"""

from __future__ import annotations

import importlib
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, overload

import ulid

from .parser import collect, expand

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from subprocess import CompletedProcess
    from typing import Any

    from .conf import Job


def iter_args(batch: str, args: str) -> Iterator[list[str]]:
    """Iterate over combinations generated from parsed arguments.

    Generate all possible combinations of arguments by parsing and
    expanding each one, yielding them as an iterator.

    Args:
        batch (str): The batch to parse.
        args (str): The arguments to parse.

    Yields:
        list[str]: a list of the parsed argument combinations.

    """
    args_ = collect(args)

    for batch_ in expand(batch):
        yield [*batch_, *args_]


def iter_batches(job: Job) -> Iterator[list[str]]:
    """Generate Hydra application arguments for a job.

    This function generates a list of Hydra application arguments
    for a given job, including the job name and the root directory
    for the sweep.

    Args:
        job (Job): The job to generate the Hydra configuration for.

    Returns:
        list[str]: A list of Hydra configuration strings.

    """
    job_name = f"hydra.job.name={job.name}"
    job_configs = shlex.split(job.with_)

    for step in job.steps:
        configs = shlex.split(step.with_) or job_configs

        for args in iter_args(step.batch, step.args):
            sweep_dir = f"hydra.sweep.dir=multirun/{ulid.ULID()}"
            yield ["--multirun", *args, job_name, sweep_dir, *configs]


@dataclass
class Task:
    """An executed task."""

    args: list[str]
    total: int
    completed: int


@dataclass
class Run(Task):
    """An executed run."""

    result: CompletedProcess


@dataclass
class Call(Task):
    """An executed call."""

    result: Any


@overload
def iter_runs(args: list[str], iterable: Iterable[list[str]]) -> Iterator[Run]: ...


@overload
def iter_runs(
    args: list[str],
    iterable: Iterable[list[str]],
    *,
    dry_run: bool = False,
) -> Iterator[Task | Run]: ...


def iter_runs(
    args: list[str],
    iterable: Iterable[list[str]],
    *,
    dry_run: bool = False,
) -> Iterator[Task | Run]:
    """Execute multiple runs of a job using shell commands."""
    executable, *args = args
    if executable == "python" and sys.platform == "win32":
        executable = sys.executable

    iterable = list(iterable)
    total = len(iterable)

    for completed, args_ in enumerate(iterable, 1):
        cmd = [executable, *args, *args_]
        if dry_run:
            yield Task(cmd, total, completed)
        else:
            result = subprocess.run(cmd, check=False)
            yield Run(cmd, total, completed, result)


@overload
def iter_calls(args: list[str], iterable: Iterable[list[str]]) -> Iterator[Call]: ...


@overload
def iter_calls(
    args: list[str],
    iterable: Iterable[list[str]],
    *,
    dry_run: bool = False,
) -> Iterator[Task | Call]: ...


def iter_calls(
    args: list[str],
    iterable: Iterable[list[str]],
    *,
    dry_run: bool = False,
) -> Iterator[Task | Call]:
    """Execute multiple calls of a job using Python functions."""
    funcname, *args = args
    func = get_callable(funcname)

    iterable = list(iterable)
    total = len(iterable)

    for completed, args_ in enumerate(iterable, 1):
        cmd = [funcname, *args, *args_]
        if dry_run:
            yield Task(cmd, total, completed)
        else:
            result = func([*args, *args_])
            yield Call(cmd, total, completed, result)


def submit(
    args: list[str],
    iterable: Iterable[list[str]],
    *,
    dry_run: bool = False,
) -> CompletedProcess | tuple[list[str], str]:
    """Submit entire job using a shell command."""
    executable, *args = args
    if executable == "python" and sys.platform == "win32":
        executable = sys.executable

    temp = NamedTemporaryFile(dir=Path.cwd(), delete=False)  # for Windows
    file = Path(temp.name)
    temp.close()

    text = "\n".join(shlex.join(args) for args in iterable)
    file.write_text(text)
    cmd = [executable, *args, file.as_posix()]

    try:
        if dry_run:
            return cmd, text
        return subprocess.run(cmd, check=False)

    finally:
        file.unlink(missing_ok=True)


def get_callable(name: str) -> Callable:
    """Get a callable from a function name."""
    if "." not in name:
        msg = f"Invalid function path: {name}."
        raise ValueError(msg)

    try:
        module_name, func_name = name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        msg = f"Failed to import or find function: {name}"
        raise ValueError(msg) from e
