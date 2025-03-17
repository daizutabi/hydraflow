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
from typing import TYPE_CHECKING

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

    total: int
    completed: int
    result: Any


@dataclass
class Run(Task):
    """An executed run."""

    result: CompletedProcess


def multirun(
    executable: str,
    args: list[str],
    iterable: Iterable[list[str]],
) -> Iterator[Run]:
    """Execute multiple runs of a job using shell commands."""
    if executable == "python" and sys.platform == "win32":
        executable = sys.executable

    iterable = list(iterable)
    total = len(iterable)

    for completed, args_ in enumerate(iterable, 1):
        result = subprocess.run([executable, *args, *args_], check=False)
        yield Run(total, completed, result)


def multicall(
    funcname: str,
    args: list[str],
    iterable: Iterable[list[str]],
) -> Iterator[Task]:
    """Execute multiple calls of a job using Python functions."""
    func = get_callable(funcname)

    iterable = list(iterable)
    total = len(iterable)

    for completed, args_ in enumerate(iterable, 1):
        result = func([*args, *args_])
        yield Task(total, completed, result)


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


def to_text(job: Job) -> str:
    """Convert the job configuration to a string.

    This function returns the job configuration for a given job.

    Args:
        job (Job): The job configuration to show.

    Returns:
        str: The job configuration.

    """
    text = ""

    it = iter_batches(job)

    if job.run:
        base_cmds = shlex.split(job.run)
        for args in it:
            cmds = " ".join([*base_cmds, *args])
            text += f"{cmds}\n"

    elif job.call:
        text = f"call: {job.call}\n"
        for args in it:
            text += f"args: {args}\n"

    return text.rstrip()
