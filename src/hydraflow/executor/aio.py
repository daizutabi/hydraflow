"""Asynchronous execution."""

from __future__ import annotations

import asyncio
from asyncio.subprocess import PIPE
from typing import TYPE_CHECKING

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

if TYPE_CHECKING:
    from asyncio.streams import StreamReader
    from collections.abc import Callable, Iterable

    from hydraflow.executor.job import Task


console = Console(log_time=False, log_path=False)


def run(iterable: Iterable[Task]) -> int | None:
    """Run multiple tasks."""
    with Progress(
        SpinnerColumn(),
        TimeElapsedColumn(),
        BarColumn(),
        TimeRemainingColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        def stdout(output: str) -> None:
            progress.log(output.rstrip())

        def stderr(output: str) -> None:
            progress.log(f"[red]{output}".rstrip())

        task_id = progress.add_task("")

        for task in iterable:
            progress.update(task_id, total=task.total)

            coro = arun(task.args, stdout, stderr)
            returncode = asyncio.run(coro)

            if returncode:
                return returncode

            progress.update(task_id, completed=task.index)

        return 0


async def arun(
    args: list[str],
    stdout: Callable[[str], None],
    stderr: Callable[[str], None],
) -> int | None:
    """Run a command asynchronously."""
    process = await asyncio.create_subprocess_exec(*args, stdout=PIPE, stderr=PIPE)
    coros = alog(process.stdout, stdout), alog(process.stderr, stderr)  # type:ignore
    await asyncio.gather(*coros)
    await process.communicate()

    return process.returncode


async def alog(reader: StreamReader, write: Callable[[str], None]) -> None:
    """Log a stream of output asynchronously."""
    while True:
        if reader.at_eof():
            break

        if out := await reader.readline():
            write(out.decode())
