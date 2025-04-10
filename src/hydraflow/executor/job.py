"""Job execution and argument handling for HydraFlow.

This module provides functionality for executing jobs in HydraFlow, including:

- Argument parsing and expansion for job parameter sets
- Batch processing of Hydra configurations
- Execution of jobs via shell commands or Python functions

The module supports two execution modes:

1. Shell command execution
2. Python function calls

Each job can consist of multiple parameter sets, and each parameter
set can have its own arguments and configurations that will be expanded
into multiple runs.
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
from typing import TYPE_CHECKING

import ulid

from .parser import collect, expand

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from subprocess import CompletedProcess
    from typing import Any

    from .conf import Job


def iter_args(each: str, all_: str) -> Iterator[list[str]]:
    """Iterate over combinations generated from parsed arguments.

    Generate all possible combinations of arguments by parsing and
    expanding each one, yielding them as an iterator.

    Args:
        each (str): The 'each' parameter to parse.
        all_ (str): The 'all' parameter to parse.

    Yields:
        list[str]: a list of the parsed argument combinations.

    """
    all_params = collect(all_)

    for each_params in expand(each):
        yield [*each_params, *all_params]


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
    job_configs = shlex.split(job.add)

    for set_ in job.sets:
        configs = shlex.split(set_.add) or job_configs

        for args in iter_args(set_.each, set_.all):
            sweep_dir = f"hydra.sweep.dir=multirun/{ulid.ULID()}"
            yield ["--multirun", *args, job_name, sweep_dir, *configs]


@dataclass
class Task:
    """A task to be executed."""

    args: list[str]
    total: int
    index: int


@dataclass
class Call(Task):
    """A call to be executed."""

    func: Callable[[], Any]


def iter_tasks(args: list[str], iterable: Iterable[list[str]]) -> Iterator[Task]:
    """Yield tasks of a job to be executed using a shell command."""
    executable, *args = args
    if executable == "python" and sys.platform == "win32":
        executable = sys.executable

    iterable = list(iterable)
    total = len(iterable)

    for index, args_ in enumerate(iterable):
        yield Task([executable, *args, *args_], total, index)


def iter_calls(args: list[str], iterable: Iterable[list[str]]) -> Iterator[Call]:
    """Yield calls of a job to be executed using a Python function."""
    funcname, *args = args
    func = get_callable(funcname)

    iterable = list(iterable)
    total = len(iterable)

    for index, args_ in enumerate(iterable):
        cmd = [funcname, *args, *args_]
        yield Call(cmd, total, index, lambda x=cmd[1:]: func(x))


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
