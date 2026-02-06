from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from tests.core.conftest import Collect, Results


def get_run_id(results: list[tuple[Path, DictConfig]], count: int) -> str:
    for path, cfg in results:
        if cfg.count == count:
            return path.parent.name
    raise ValueError


@pytest.fixture(scope="module")
def results(collect: Collect, tmp_path_factory: pytest.TempPathFactory) -> Results:
    cwd = tmp_path_factory.mktemp("mlflow_skip_finished")

    running = RunStatus.to_string(RunStatus.RUNNING)  # pyright: ignore[reportUnknownMemberType]

    file = Path(__file__).parent / "skip_finished.py"
    args = ["-m", "count=1,2,3"]

    results = collect(file, args, cwd=cwd)
    db = str(cwd.joinpath("mlflow.db"))
    client = MlflowClient(f"sqlite:///{db}")
    client.set_terminated(get_run_id(results, 2), status=running)
    client.set_terminated(get_run_id(results, 3), status=running)
    results = collect(file, args, cwd=cwd)
    client.set_terminated(get_run_id(results, 3), status=running)
    return collect(file, args, cwd=cwd)


def test_len(results: Results) -> None:
    assert len(results) == 3


@pytest.fixture(scope="module", params=range(3))
def result(results: Results, request: pytest.FixtureRequest):
    assert isinstance(request.param, int)
    return results[request.param]


@pytest.fixture(scope="module")
def path(result: tuple[Path, DictConfig]):
    return result[0]


@pytest.fixture(scope="module")
def cfg(result: tuple[Path, DictConfig]):
    return result[1]


@pytest.fixture(scope="module")
def count(cfg: DictConfig):
    return cfg.count


@pytest.fixture(scope="module")
def text(path: Path):
    return path.joinpath("a.txt").read_text()


def test_count(text: str, count: int) -> None:
    assert len(text.splitlines()) == count


def test_config(text: str, count: int) -> None:
    assert int(text.split(" ", maxsplit=1)[0]) == count


def test_run(text: str, path: Path) -> None:
    line = text.splitlines()[-1]
    assert line.split(" ", maxsplit=1)[1] == path.parent.name
