from pathlib import Path

import mlflow
import pytest

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(scope="module")
def root_dir(chdir):
    return Path("mlruns").absolute()


@pytest.fixture(scope="module", autouse=True)
def setup(chdir):
    mlflow.set_experiment("e1")
    with mlflow.start_run():
        mlflow.log_text("1", "text.txt")
    with mlflow.start_run():
        mlflow.log_text("2", "text.txt")
    mlflow.set_experiment("e2")
    with mlflow.start_run():
        mlflow.log_text("3", "text.txt")
    with mlflow.start_run():
        mlflow.log_text("4", "text.txt")
    with mlflow.start_run():
        mlflow.log_text("5", "text.txt")


def test_iter_experiment_dirs(root_dir: Path):
    from hydraflow.core.io import get_experiment_name, iter_experiment_dirs

    names = [get_experiment_name(p) for p in iter_experiment_dirs(root_dir)]
    assert sorted(names) == ["e1", "e2"]  # type: ignore


@pytest.mark.parametrize(
    ("e", "es"),
    [("e1", ["e1"]), ("e*", ["e1", "e2"]), ("*", ["e1", "e2"]), ("*2", ["e2"])],
)
def test_iter_experiment_dirs_glob(root_dir: Path, e, es):
    from hydraflow.core.io import get_experiment_name, iter_experiment_dirs

    names = [get_experiment_name(p) for p in iter_experiment_dirs(root_dir, e)]
    assert sorted(names) == es  # type: ignore


def test_iter_experiment_dirs_filter(root_dir: Path):
    from hydraflow.core.io import get_experiment_name, iter_experiment_dirs

    it = iter_experiment_dirs(root_dir, experiment_names="e1")
    assert [get_experiment_name(p) for p in it] == ["e1"]


def test_iter_experiment_dirs_filter_callable(root_dir: Path):
    from hydraflow.core.io import get_experiment_name, iter_experiment_dirs

    it = iter_experiment_dirs(root_dir, experiment_names=lambda name: name == "e2")
    assert [get_experiment_name(p) for p in it] == ["e2"]


def test_predicate_experiment_dir():
    from hydraflow.core.io import predicate_experiment_dir

    assert predicate_experiment_dir(Path()) is False


def test_get_experiment_name_none(root_dir: Path):
    from hydraflow.core.io import get_experiment_name

    assert get_experiment_name(root_dir.parent) is None


def test_get_experiment_name_metafile_none(root_dir: Path):
    from hydraflow.core.io import get_experiment_name

    (root_dir / "meta.yaml").touch()
    assert get_experiment_name(root_dir) is None


def test_iter_run_dirs(root_dir: Path):
    from hydraflow.core.io import iter_run_dirs

    assert len(list(iter_run_dirs(root_dir))) == 5


def test_iter_artifacts_dirs(root_dir: Path):
    from hydraflow.core.io import iter_artifacts_dirs

    assert len(list(iter_artifacts_dirs(root_dir))) == 5


def test_iter_artifact_paths(root_dir: Path):
    from hydraflow.core.io import iter_artifact_paths

    it = iter_artifact_paths(root_dir, "text.txt")
    text = sorted("".join(p.read_text() for p in it))
    assert text == ["1", "2", "3", "4", "5"]
