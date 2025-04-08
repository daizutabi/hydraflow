from pathlib import Path

import pytest
from omegaconf import DictConfig

pytestmark = pytest.mark.xdist_group(name="group5")


@pytest.fixture(scope="module")
def results(collect):
    file = Path(__file__).parent / "default.py"
    collect(file, ["-m", "count=1,2"])
    return collect(file, ["-m", "name=a", "count=1,2,3,4"])


def test_len(results):
    assert len(results) == 4


@pytest.fixture(scope="module", params=range(4))
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
    return int(cfg.count)


@pytest.fixture(scope="module")
def text(path: Path):
    return path.joinpath("a.txt").read_text()


def test_count(text: str, count: int):
    assert text == str(count)


@pytest.fixture(scope="module")
def cwd(path: Path):
    return Path(path.joinpath("b.txt").read_text())


def test_cwd(cwd: Path, experiment_name: str):
    assert cwd.name == experiment_name
