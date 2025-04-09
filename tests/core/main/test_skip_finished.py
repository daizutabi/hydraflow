from pathlib import Path

import pytest
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig


def get_run_id(results: list[tuple[Path, DictConfig]], count: int) -> str:
    for path, cfg in results:
        if cfg.count == count:
            return path.parent.name
    raise ValueError


@pytest.fixture(scope="module")
def results(collect):
    client = MlflowClient()
    running = RunStatus.to_string(RunStatus.RUNNING)

    file = Path(__file__).parent / "skip_finished.py"
    args = ["-m", "count=1,2,3"]

    results = collect(file, args)
    client.set_terminated(get_run_id(results, 2), status=running)
    client.set_terminated(get_run_id(results, 3), status=running)
    results = collect(file, args)
    client.set_terminated(get_run_id(results, 3), status=running)
    return collect(file, args)


def test_len(results):
    assert len(results) == 3


@pytest.fixture(scope="module", params=range(3))
def result(results, request: pytest.FixtureRequest):
    return results[request.param]


@pytest.fixture(scope="module")
def path(result):
    return result[0]


@pytest.fixture(scope="module")
def cfg(result):
    return result[1]


@pytest.fixture(scope="module")
def count(cfg: DictConfig):
    return cfg.count


@pytest.fixture(scope="module")
def text(path: Path):
    return path.joinpath("a.txt").read_text()


def test_count(text: str, count: int):
    assert len(text.splitlines()) == count


def test_config(text: str, count: int):
    assert int(text.split(" ", maxsplit=1)[0]) == count


def test_run(text: str, path: Path):
    line = text.splitlines()[-1]
    assert line.split(" ", maxsplit=1)[1] == path.parent.name
