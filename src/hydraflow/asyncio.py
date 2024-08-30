from __future__ import annotations

import asyncio
import logging
import sys
from asyncio.subprocess import PIPE
from pathlib import Path
from typing import TYPE_CHECKING

import watchfiles

if TYPE_CHECKING:
    from asyncio.streams import StreamReader
    from collections.abc import Callable

    from watchfiles import Change


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def execute_command(
    program: str,
    *args: str,
    stdout: Callable[[str], None] | None = None,
    stderr: Callable[[str], None] | None = None,
    stop_event: asyncio.Event,
) -> int:
    """
    Runs a command asynchronously and pass the output to callback functions.

    Args:
        program (str): The program to run.
        *args (str): Arguments for the program.
        stdout (Callable[[str], None] | None): Callback for standard output.
        stderr (Callable[[str], None] | None): Callback for standard error.
        stop_event (asyncio.Event): Event to signal when the process is done.

    Returns:
        int: The return code of the process.
    """
    try:
        process = await asyncio.create_subprocess_exec(program, *args, stdout=PIPE, stderr=PIPE)
        await asyncio.gather(
            process_stream(process.stdout, stdout),
            process_stream(process.stderr, stderr),
        )
        returncode = await process.wait()

    except Exception as e:
        logger.error(f"Error running command: {e}")
        returncode = 1

    finally:
        stop_event.set()

    return returncode


async def process_stream(
    stream: StreamReader | None,
    callback: Callable[[str], None] | None,
) -> None:
    """
    Reads a stream asynchronously and pass each line to a callback function.

    Args:
        stream (StreamReader | None): The stream to read from.
        callback (Callable[[str], None] | None): The callback function to handle
        each line.
    """
    if stream is None or callback is None:
        return

    while True:
        line = await stream.readline()
        if line:
            callback(line.decode().strip())
        else:
            break


async def monitor_file_changes(
    paths: list[str | Path],
    callback: Callable[[set[tuple[Change, str]]], None],
    stop_event: asyncio.Event,
) -> None:
    """
    Watches for file changes in specified paths and pass the changes to a
    callback function.

    Args:
        paths (list[str | Path]): List of paths to monitor for changes.
        callback (Callable[[set[tuple[Change, str]]], None]): The callback
        function to handle file changes.
        stop_event (asyncio.Event): Event to signal when to stop watching.
    """
    str_paths = [str(path) for path in paths]
    try:
        async for changes in watchfiles.awatch(*str_paths, debug=True, stop_event=stop_event):
            callback(changes)
    except Exception as e:
        logger.error(f"Error watching files: {e}")


async def run_and_monitor(
    program: str,
    *args: str,
    stdout: Callable[[str], None] | None = None,
    stderr: Callable[[str], None] | None = None,
    watch: Callable[[set[tuple[Change, str]]], None] | None = None,
    paths: list[str | Path] | None = None,
) -> int:
    """
    Runs a command and optionally watch for file changes concurrently.

    Args:
        program (str): The program to run.
        *args (str): Arguments for the program.
        stdout (Callable[[str], None] | None): Callback for standard output.
        stderr (Callable[[str], None] | None): Callback for standard error.
        watch (Callable[[set[tuple[Change, str]]], None] | None): Callback for
        file changes.
        paths (list[str | Path] | None): List of paths to monitor for changes.
    """
    stop_event = asyncio.Event()
    run_task = asyncio.create_task(
        execute_command(program, *args, stop_event=stop_event, stdout=stdout, stderr=stderr)
    )
    if watch and paths:
        monitor_task = asyncio.create_task(monitor_file_changes(paths, watch, stop_event))
    else:
        monitor_task = None

    try:
        if monitor_task:
            await asyncio.gather(run_task, monitor_task)
        else:
            await run_task

    except Exception as e:
        logger.error(f"Error in run_and_monitor: {e}")
    finally:
        stop_event.set()
        await run_task
        if monitor_task:
            await monitor_task

    return run_task.result()


def run(
    program: str,
    *args: str,
    stdout: Callable[[str], None] | None = None,
    stderr: Callable[[str], None] | None = None,
    watch: Callable[[set[tuple[Change, str]]], None] | None = None,
    paths: list[str | Path] | None = None,
) -> int:
    """
    Runs a command and optionally watch for file changes concurrently.

    Args:
        program (str): The program to run.
        *args (str): Arguments for the program.
        stdout (Callable[[str], None] | None): Callback for standard output.
        stderr (Callable[[str], None] | None): Callback for standard error.
        watch (Callable[[set[tuple[Change, str]]], None] | None): Callback for
        file changes.
        paths (list[str | Path] | None): List of paths to monitor for changes.
    """
    return asyncio.run(
        run_and_monitor(program, *args, stdout=stdout, stderr=stderr, watch=watch, paths=paths)
    )


if __name__ == "__main__":

    def stdout(line: str) -> None:
        logger.info(f"STDOUT: {line}")

    def stderr(line: str) -> None:
        logger.error(f"STDERR: {line}")

    def watch(changes: set[tuple[Change, str]]):
        logger.info(f"File changes detected: {changes}")

    run(
        sys.executable,
        "a.py",
        stdout=stdout,
        stderr=stderr,
        watch=watch,
        paths=["."],
    )
