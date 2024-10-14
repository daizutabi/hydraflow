from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from threading import Thread
from time import sleep
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from watchfiles import Change


def sleep_write(path: Path):
    sleep(0.1)
    path.write_text("hello")


@pytest.fixture
def write_soon():
    threads = []

    def start(path: Path):
        thread = Thread(target=sleep_write, args=(path,))
        thread.start()
        threads.append(thread)

    yield start

    for t in threads:
        t.join()


@pytest.mark.asyncio
async def test_execute_command():
    from hydraflow.asyncio import execute_command

    stdout_lines = []
    stderr_lines = []
    stop_event = asyncio.Event()

    def stdout(line: str) -> None:
        stdout_lines.append(line)

    def stderr(line: str) -> None:
        stderr_lines.append(line)

    c = "import sys;print('hello');print('world', file=sys.stderr);sys.exit(100)"
    return_code = await execute_command(
        sys.executable,
        "-c",
        c,
        stdout=stdout,
        stderr=stderr,
        stop_event=stop_event,
    )

    assert return_code == 100
    assert stdout_lines == ["hello"]
    assert stderr_lines == ["world"]
    assert stop_event.is_set()


@pytest.mark.asyncio
async def test_monitor_file_changes(tmp_path: Path, write_soon: Callable[[Path], None]):
    from hydraflow.asyncio import monitor_file_changes

    changes_detected = []
    stop_event = asyncio.Event()

    def callback(changes: set[tuple[Change, str]]) -> None:
        changes_detected.extend(changes)

    write_soon(tmp_path / "test.txt")
    monitor_task = asyncio.create_task(
        monitor_file_changes([tmp_path], callback, stop_event),
    )

    await asyncio.sleep(1)
    stop_event.set()
    await monitor_task
    await asyncio.sleep(1)

    assert len(changes_detected) > 0


@pytest.mark.asyncio
async def test_run_and_monitor(tmp_path: Path):
    from hydraflow.asyncio import run_and_monitor

    stdout_lines = []
    stderr_lines = []
    changes_detected = []

    def stdout(line: str) -> None:
        stdout_lines.append(line)

    def stderr(line: str) -> None:
        stderr_lines.append(line)

    def watch(changes: set[tuple[Change, str]]) -> None:
        changes_detected.extend(changes)

    path = (tmp_path / "test.txt").as_posix()
    c = "import sys;print('hello');print('world', file=sys.stderr);"
    c += f"import pathlib;pathlib.Path('{path}').write_text('hello world');"
    c += "import time;time.sleep(1);sys.exit(200)"

    return_code = await run_and_monitor(
        sys.executable,
        "-c",
        c,
        stdout=stdout,
        stderr=stderr,
        watch=watch,
        paths=[tmp_path],
    )

    assert return_code == 200
    assert stdout_lines == ["hello"]
    assert stderr_lines == ["world"]
    assert Path(path).read_text() == "hello world"
    assert len(changes_detected) >= 1


def test_run(tmp_path: Path):
    from hydraflow.asyncio import run

    stdout_lines = []
    stderr_lines = []
    changes_detected = []

    def stdout(line: str) -> None:
        stdout_lines.append(line)

    def stderr(line: str) -> None:
        stderr_lines.append(line)

    def watch(changes: set[tuple[Change, str]]) -> None:
        changes_detected.extend(changes)

    path = (tmp_path / "test").as_posix()
    c = "import sys;print('hello');print('world', file=sys.stderr);"
    c += f"import pathlib;pathlib.Path('{path}').write_text('hello world');"
    c += f"pathlib.Path('{path}2').write_text('hello world2');"
    c += "import time;time.sleep(1);sys.exit(100)"

    return_code = run(
        sys.executable,
        "-c",
        c,
        stdout=stdout,
        stderr=stderr,
        watch=watch,
        paths=[tmp_path],
    )

    assert return_code == 100
    assert stdout_lines == ["hello"]
    assert stderr_lines == ["world"]
    assert Path(path).read_text() == "hello world"
    assert len(changes_detected) >= 2


@pytest.mark.asyncio
async def test_execute_command_nonexistent():
    from hydraflow.asyncio import execute_command

    stop_event = asyncio.Event()

    rc = await execute_command("nonexistent_command", stop_event=stop_event)
    assert rc == 1
    assert stop_event.is_set()


@pytest.mark.asyncio
async def test_process_stream_none():
    from hydraflow.asyncio import process_stream

    assert await process_stream(None, None) is None


@pytest.mark.asyncio
async def test_monitor_file_changes_error():
    from hydraflow.asyncio import monitor_file_changes

    stop_event = asyncio.Event()

    with pytest.raises(FileNotFoundError):
        await monitor_file_changes(["nonexistent_path"], lambda _: None, stop_event)


@pytest.mark.asyncio
async def test_run_and_monitor_none():
    from hydraflow.asyncio import run_and_monitor

    assert await run_and_monitor("echo", "hello") == 0


@pytest.mark.asyncio
async def test_run_and_monitor_error():
    from hydraflow.asyncio import run_and_monitor

    with pytest.raises(FileNotFoundError):
        await run_and_monitor(
            "nonexistent_command",
            watch=lambda _: None,
            paths=["nonexistent_path"],
        )


def test_run_cwd():
    from hydraflow.asyncio import run

    return_code = run(sys.executable, "--version", watch=lambda _: None)

    assert return_code == 0
