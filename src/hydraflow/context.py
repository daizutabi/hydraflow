from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
from hydra.core.hydra_config import HydraConfig
from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from hydraflow.mlflow import log_params
from hydraflow.run import get_artifact_path
from hydraflow.util import uri_to_path

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from mlflow.entities.run import Run
    from pandas import Series


@dataclass
class Info:
    output_dir: Path
    artifact_dir: Path


@contextmanager
def log_run(
    config: object,
    *,
    synchronous: bool | None = None,
) -> Iterator[Info]:
    log_params(config, synchronous=synchronous)

    hc = HydraConfig.get()
    output_dir = Path(hc.runtime.output_dir)
    uri = mlflow.get_artifact_uri()
    location = Info(output_dir, uri_to_path(uri))

    # Save '.hydra' config directory first.
    output_subdir = output_dir / (hc.output_subdir or "")
    mlflow.log_artifacts(output_subdir.as_posix(), hc.output_subdir)

    try:
        yield location

    finally:
        # Save output_dir including '.hydra' config directory.
        mlflow.log_artifacts(output_dir.as_posix())


@contextmanager
def watch(
    func: Callable[[Path], None],
    dir: Path | str = "",
    timeout: int = 600,
) -> Iterator[None]:
    if not dir:
        uri = mlflow.get_artifact_uri()
        dir = uri_to_path(uri)

    handler = Handler(func)
    observer = Observer()
    observer.schedule(handler, dir, recursive=True)
    observer.start()

    try:
        yield

    finally:
        elapsed = 0
        while not observer.event_queue.empty():
            time.sleep(0.2)
            elapsed += 0.2
            if elapsed > timeout:
                break

        observer.stop()
        observer.join()


class Handler(FileSystemEventHandler):
    def __init__(self, func: Callable[[Path], None]) -> None:
        self.func = func

    def on_modified(self, event: FileModifiedEvent) -> None:
        file = Path(event.src_path)
        if file.is_file():
            self.func(file)


@contextmanager
def chdir_artifact(
    run: Run | Series | str,
    artifact_path: str | None = None,
) -> Iterator[Path]:
    curdir = Path.cwd()

    artifact_dir = get_artifact_path(run, artifact_path)

    os.chdir(artifact_dir)
    try:
        yield artifact_dir
    finally:
        os.chdir(curdir)
