"""Provide utility functions for HydraFlow."""

from __future__ import annotations

import fnmatch
import urllib.parse
import urllib.request
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from mlflow.entities import Experiment, Run


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
    uri = run.info.artifact_uri  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

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


def predicate_experiment(
    experiment: Experiment,
    experiment_names: list[str] | Callable[[str], bool] | None = None,
) -> bool:
    """Predicate an experiment based on the experiment names."""
    if experiment_names is None:
        return True

    name = cast("str", experiment.name)
    if isinstance(experiment_names, list):
        return any(fnmatch.fnmatch(name, e) for e in experiment_names)

    return experiment_names(name)


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
    import mlflow

    es = mlflow.search_experiments(filter_string='name != "Default"')
    for experiment in es:
        loc = cast("str", experiment.artifact_location)
        if Path(file_uri_to_path(loc)) == experiment_dir:
            return cast("str", experiment.name)

    return ""


def get_experiment_names() -> list[str]:
    """Get the experiment names.

    Returns:
        list[str]: A list of experiment names sorted by the name.

    """
    import mlflow

    es = mlflow.search_experiments(filter_string='name != "Default"')
    return [e.name for e in es]  # pyright: ignore[reportUnknownMemberType]


def iter_experiment_dirs(
    experiment_names: str | list[str] | Callable[[str], bool] | None = None,
) -> Iterator[Path]:
    """Iterate over the experiment directories."""  # noqa: DOC402
    import mlflow

    if isinstance(experiment_names, str):
        experiment_names = [experiment_names]

    es = mlflow.search_experiments(filter_string='name != "Default"')
    for experiment in es:
        if predicate_experiment(experiment, experiment_names):
            loc = cast("str", experiment.artifact_location)
            yield Path(file_uri_to_path(loc))


def iter_run_dirs(
    experiment_names: str | list[str] | Callable[[str], bool] | None = None,
) -> Iterator[Path]:
    """Iterate over the run directories in the tracking directory."""  # noqa: DOC402
    for experiment_dir in iter_experiment_dirs(experiment_names):
        for path in experiment_dir.iterdir():
            if path.is_dir() and (path / "artifacts").exists():
                yield path


def iter_artifacts_dirs(
    experiment_names: str | list[str] | Callable[[str], bool] | None = None,
) -> Iterator[Path]:
    """Iterate over the artifacts directories in the tracking directory."""  # noqa: DOC402
    for path in iter_run_dirs(experiment_names):
        yield path / "artifacts"


def iter_artifact_paths(
    artifact_path: str | Path,
    experiment_names: str | list[str] | Callable[[str], bool] | None = None,
) -> Iterator[Path]:
    """Iterate over the artifact paths in the tracking directory."""  # noqa: DOC402
    for path in iter_artifacts_dirs(experiment_names):
        yield path / artifact_path
