"""Provide utility functions for HydraFlow."""

from __future__ import annotations

import fnmatch
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from mlflow.entities import Run


def file_uri_to_path(uri: str) -> Path:
    """Convert a file URI to a local path."""
    if not uri.startswith("file:"):
        return Path(uri)

    path = urllib.parse.urlparse(uri).path
    return Path(urllib.request.url2pathname(path))  # for Windows


def get_artifact_dir(run: Run) -> Path:
    """Retrieve the artifact directory for the given run.

    This function uses MLflow to get the artifact directory for the given run.

    Args:
        run (Run | None): The run object. Defaults to None.

    Returns:
        The local path to the directory where the artifacts are downloaded.

    """
    uri = run.info.artifact_uri

    if not isinstance(uri, str):
        raise NotImplementedError

    return file_uri_to_path(uri)


def log_text(run: Run, from_dir: Path, pattern: str = "*.log") -> None:
    """Log text files in the given directory as artifacts.

    Append the text files to the existing text file in the artifact directory.

    Args:
        run (Run): The run object.
        from_dir (Path): The directory to find the logs in.
        pattern (str): The pattern to match the logs.

    """
    import mlflow

    artifact_dir = get_artifact_dir(run)

    for file in from_dir.glob(pattern):
        if not file.is_file():
            continue

        file_artifact = artifact_dir / file.name
        if file_artifact.exists():
            text = file_artifact.read_text()
            if not text.endswith("\n"):
                text += "\n"
        else:
            text = ""

        text += file.read_text()
        mlflow.log_text(text, file.name)


def get_experiment_name(path: Path) -> str | None:
    """Get the experiment name from the meta file."""
    metafile = path / "meta.yaml"
    if not metafile.exists():
        return None
    lines = metafile.read_text().splitlines()
    for line in lines:
        if line.startswith("name:"):
            return line.split(":")[1].strip()
    return None


def predicate_experiment_dir(
    path: Path,
    experiment_names: list[str] | Callable[[str], bool] | None = None,
) -> bool:
    """Predicate an experiment directory based on the path and experiment names."""
    if not path.is_dir() or path.name in [".trash", "0"]:
        return False

    name = get_experiment_name(path)
    if not name:
        return False

    if experiment_names is None:
        return True

    if isinstance(experiment_names, list):
        return any(fnmatch.fnmatch(name, e) for e in experiment_names)

    return experiment_names(name)


def iter_experiment_dirs(
    root_dir: str | Path,
    experiment_names: str | list[str] | Callable[[str], bool] | None = None,
) -> Iterator[Path]:
    """Iterate over the experiment directories in the root directory."""
    if isinstance(experiment_names, str):
        experiment_names = [experiment_names]

    for path in Path(root_dir).iterdir():
        if predicate_experiment_dir(path, experiment_names):
            yield path


def iter_run_dirs(
    root_dir: str | Path,
    experiment_names: str | list[str] | Callable[[str], bool] | None = None,
) -> Iterator[Path]:
    """Iterate over the run directories in the root directory."""
    for experiment_dir in iter_experiment_dirs(root_dir, experiment_names):
        for path in experiment_dir.iterdir():
            if path.is_dir() and (path / "artifacts").exists():
                yield path


def iter_artifacts_dirs(
    root_dir: str | Path,
    experiment_names: str | list[str] | Callable[[str], bool] | None = None,
) -> Iterator[Path]:
    """Iterate over the artifacts directories in the root directory."""
    for path in iter_run_dirs(root_dir, experiment_names):
        yield path / "artifacts"


def iter_artifact_paths(
    root_dir: str | Path,
    artifact_path: str | Path,
    experiment_names: str | list[str] | Callable[[str], bool] | None = None,
) -> Iterator[Path]:
    """Iterate over the artifact paths in the root directory."""
    for path in iter_artifacts_dirs(root_dir, experiment_names):
        yield path / artifact_path
