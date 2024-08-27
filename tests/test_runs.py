from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from mlflow.entities import Run


@pytest.fixture
def runs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    mlflow.set_experiment("test_run")
    for x in range(6):
        with mlflow.start_run(run_name=f"{x}"):
            mlflow.log_param("p", x)
            mlflow.log_param("q", 0)
            mlflow.log_text(f"{x}", "abc.txt")

    x = mlflow.search_runs(output_format="list", order_by=["params.p"])
    assert isinstance(x, list)
    assert isinstance(x[0], Run)
    return x


def test_filter_one(runs: list[Run]):
    from hydraflow.runs import filter_runs

    assert len(runs) == 6
    x = filter_runs(runs, {"p": 1})
    assert len(x) == 1


def test_filter_all(runs: list[Run]):
    from hydraflow.runs import filter_runs

    assert len(runs) == 6
    x = filter_runs(runs, {"q": 0})
    assert len(x) == 6


def test_get_run(runs: list[Run]):
    from hydraflow.runs import get_run

    run = get_run(runs, {"p": 4})
    assert isinstance(run, Run)
    assert run.data.params["p"] == "4"


def test_get_error(runs: list[Run]):
    from hydraflow.runs import get_run

    with pytest.raises(ValueError):
        get_run(runs, {"q": 0})


def test_get_param_names(runs: list[Run]):
    from hydraflow.runs import get_param_names

    params = get_param_names(runs)
    assert len(params) == 2
    assert "p" in params
    assert "q" in params


def test_get_param_dict(runs: list[Run]):
    from hydraflow.runs import get_param_dict

    params = get_param_dict(runs)
    assert len(params["p"]) == 6
    assert len(params["q"]) == 1


@pytest.mark.parametrize("i", range(6))
def test_chdir_artifact_list(i: int, runs: list[Run]):
    from hydraflow.context import chdir_artifact

    with chdir_artifact(runs[i]):
        assert Path("abc.txt").read_text() == f"{i}"

    assert not Path("abc.txt").exists()


# def test_hydra_output_dir_error(runs_list: list[Run]):
#     from hydraflow.runs import get_hydra_output_dir

#     with pytest.raises(FileNotFoundError):
#         get_hydra_output_dir(runs_list[0])


def test_runs_repr(runs):
    from hydraflow.runs import Runs

    assert repr(Runs(runs)) == "Runs(6)"


def test_runs_filter(runs):
    from hydraflow.runs import Runs

    runs = Runs(runs)

    assert len(runs.filter({})) == 6
    assert len(runs.filter({"p": 1})) == 1
    assert len(runs.filter({"q": 0})) == 6
    assert len(runs.filter({"q": -1})) == 0


def test_runs_get(runs):
    from hydraflow.runs import Run, Runs

    runs = Runs(runs)
    run = runs.get({"p": 4})
    assert isinstance(run, Run)


def test_runs_get_params_names(runs):
    from hydraflow.runs import Runs

    runs = Runs(runs)
    names = runs.get_param_names()
    assert len(names) == 2
    assert "p" in names
    assert "q" in names


def test_runs_get_params_dict(runs):
    from hydraflow.runs import Runs

    runs = Runs(runs)
    params = runs.get_param_dict()
    assert params["p"] == ["0", "1", "2", "3", "4", "5"]
    assert params["q"] == ["0"]
