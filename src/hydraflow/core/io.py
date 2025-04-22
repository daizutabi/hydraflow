"""Provide utility functions for HydraFlow."""

from __future__ import annotations

import fnmatch
import urllib.parse
import urllib.request
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

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
        run (Run | None): The run instance. Defaults to None.

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
        run (Run): The run instance.
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


@cache
def get_experiment_name(experiment_dir: Path) -> str:
    """Get the job name from an experiment directory.

    Extracts the job name from the meta.yaml file. Returns an empty string
    if the file does not exist or if the job name cannot be found.

    Args:
        experiment_dir: Path to the experiment directory containing the meta.yaml file

    Returns:
        The job name as a string, or an empty string if the file does not exist

    """
    path = experiment_dir / "meta.yaml"
    if not path.exists():
        return ""

    meta = OmegaConf.load(experiment_dir / "meta.yaml")
    return OmegaConf.select(meta, "name", default="")


def predicate_experiment_dir(
    experiment_dir: Path,
    experiment_names: list[str] | Callable[[str], bool] | None = None,
) -> bool:
    """Predicate an experiment directory based on the path and experiment names."""
    if not experiment_dir.is_dir() or experiment_dir.name in [".trash", "0"]:
        return False

    name = get_experiment_name(experiment_dir)
    if not name:
        return False

    if experiment_names is None:
        return True

    if isinstance(experiment_names, list):
        return any(fnmatch.fnmatch(name, e) for e in experiment_names)

    return experiment_names(name)


def get_experiment_names(tracking_dir: str | Path) -> list[str]:
    """Get the experiment names from the tracking directory.

    Returns:
        list[str]: A list of experiment names sorted by the name.

    """
    names = [get_experiment_name(path) for path in Path(tracking_dir).iterdir()]
    return sorted(name for name in names if name and name != "Default")


def iter_experiment_dirs(
    tracking_dir: str | Path,
    experiment_names: str | list[str] | Callable[[str], bool] | None = None,
) -> Iterator[Path]:
    """Iterate over the experiment directories in the tracking directory."""
    if isinstance(experiment_names, str):
        experiment_names = [experiment_names]

    for path in Path(tracking_dir).iterdir():
        if predicate_experiment_dir(path, experiment_names):
            yield path


def iter_run_dirs(
    tracking_dir: str | Path,
    experiment_names: str | list[str] | Callable[[str], bool] | None = None,
) -> Iterator[Path]:
    """Iterate over the run directories in the tracking directory."""
    for experiment_dir in iter_experiment_dirs(tracking_dir, experiment_names):
        for path in experiment_dir.iterdir():
            if path.is_dir() and (path / "artifacts").exists():
                yield path


def iter_artifacts_dirs(
    tracking_dir: str | Path,
    experiment_names: str | list[str] | Callable[[str], bool] | None = None,
) -> Iterator[Path]:
    """Iterate over the artifacts directories in the tracking directory."""
    for path in iter_run_dirs(tracking_dir, experiment_names):
        yield path / "artifacts"


def iter_artifact_paths(
    tracking_dir: str | Path,
    artifact_path: str | Path,
    experiment_names: str | list[str] | Callable[[str], bool] | None = None,
) -> Iterator[Path]:
    """Iterate over the artifact paths in the tracking directory."""
    for path in iter_artifacts_dirs(tracking_dir, experiment_names):
        yield path / artifact_path
