from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from mlflow.entities import Run

from hydraflow.runs import Runs


@pytest.fixture
def runs(monkeypatch, tmp_path):
    from hydraflow.runs import search_runs

    monkeypatch.chdir(tmp_path)

    mlflow.set_experiment("test_run")
    for x in range(6):
        with mlflow.start_run(run_name=f"{x}"):
            mlflow.log_param("p", x)
            mlflow.log_param("q", 0)
            mlflow.log_param("r", x % 3)
            mlflow.log_text(f"{x}", "abc.txt")

    x = search_runs()
    assert isinstance(x, Runs)
    return x


@pytest.fixture
def run_list(runs: Runs):
    return runs._runs


def test_search_runs_sorted(run_list: list[Run]):
    assert [run.data.params["p"] for run in run_list] == ["0", "1", "2", "3", "4", "5"]


def test_filter_none(run_list: list[Run]):
    from hydraflow.runs import filter_runs

    assert run_list == filter_runs(run_list)


def test_filter_one(run_list: list[Run]):
    from hydraflow.runs import filter_runs

    assert len(run_list) == 6
    x = filter_runs(run_list, {"p": 1})
    assert len(x) == 1
    x = filter_runs(run_list, p=1)
    assert len(x) == 1


def test_filter_all(run_list: list[Run]):
    from hydraflow.runs import filter_runs

    assert len(run_list) == 6
    x = filter_runs(run_list, {"q": 0})
    assert len(x) == 6
    x = filter_runs(run_list, q=0)
    assert len(x) == 6


def test_filter_invalid_param(run_list: list[Run]):
    from hydraflow.runs import filter_runs

    x = filter_runs(run_list, {"invalid": 0})
    assert len(x) == 6


def test_find_run(run_list: list[Run]):
    from hydraflow.runs import find_run

    x = find_run(run_list, {"r": 1})
    assert isinstance(x, Run)
    assert x.data.params["p"] == "1"
    x = find_run(run_list, r=2)
    assert isinstance(x, Run)
    assert x.data.params["p"] == "2"


def test_find_last_run(run_list: list[Run]):
    from hydraflow.runs import find_last_run

    x = find_last_run(run_list, {"r": 1})
    assert isinstance(x, Run)
    assert x.data.params["p"] == "4"
    x = find_last_run(run_list, r=2)
    assert isinstance(x, Run)
    assert x.data.params["p"] == "5"


def test_get_run(run_list: list[Run]):
    from hydraflow.runs import get_run

    run = get_run(run_list, {"p": 4})
    assert isinstance(run, Run)
    assert run.data.params["p"] == "4"


def test_get_error(run_list: list[Run]):
    from hydraflow.runs import get_run

    with pytest.raises(ValueError):
        get_run(run_list, {"q": 0})


def test_get_param_names(run_list: list[Run]):
    from hydraflow.runs import get_param_names

    params = get_param_names(run_list)
    assert len(params) == 3
    assert "p" in params
    assert "q" in params
    assert "r" in params


def test_get_param_dict(run_list: list[Run]):
    from hydraflow.runs import get_param_dict

    params = get_param_dict(run_list)
    assert len(params["p"]) == 6
    assert len(params["q"]) == 1


@pytest.mark.parametrize("i", range(6))
def test_chdir_artifact_list(i: int, run_list: list[Run]):
    from hydraflow.context import chdir_artifact

    with chdir_artifact(run_list[i]):
        assert Path("abc.txt").read_text() == f"{i}"

    assert not Path("abc.txt").exists()


# def test_hydra_output_dir_error(runs_list: list[Run]):
#     from hydraflow.runs import get_hydra_output_dir

#     with pytest.raises(FileNotFoundError):
#         get_hydra_output_dir(runs_list[0])


def test_runs_repr(runs: Runs):
    assert repr(runs) == "Runs(6)"


def test_runs_filter(runs: Runs):
    assert len(runs.filter()) == 6
    assert len(runs.filter({})) == 6
    assert len(runs.filter({"p": 1})) == 1
    assert len(runs.filter({"q": 0})) == 6
    assert len(runs.filter({"q": -1})) == 0
    assert len(runs.filter(p=5)) == 1
    assert len(runs.filter(q=0)) == 6
    assert len(runs.filter(q=-1)) == 0
    assert len(runs.filter({"r": 2})) == 2
    assert len(runs.filter(r=0)) == 2


def test_runs_get(runs: Runs):
    from hydraflow.runs import Run

    run = runs.get({"p": 4})
    assert isinstance(run, Run)
    run = runs.get(p=2)
    assert isinstance(run, Run)


def test_runs_get_params_names(runs: Runs):
    names = runs.get_param_names()
    assert len(names) == 3
    assert "p" in names
    assert "q" in names
    assert "r" in names


def test_runs_get_params_dict(runs: Runs):
    params = runs.get_param_dict()
    assert params["p"] == ["0", "1", "2", "3", "4", "5"]
    assert params["q"] == ["0"]
    assert params["r"] == ["0", "1", "2"]


def test_runs_find(runs: Runs):
    from hydraflow.runs import Run

    run = runs.find({"r": 0})
    assert isinstance(run, Run)
    assert run.data.params["p"] == "0"
    run = runs.find(r=2)
    assert isinstance(run, Run)
    assert run.data.params["p"] == "2"


def test_runs_find_none(runs: Runs):
    run = runs.find({"r": 10})
    assert run is None


def test_runs_find_last(runs: Runs):
    from hydraflow.runs import Run

    run = runs.find_last({"r": 0})
    assert isinstance(run, Run)
    assert run.data.params["p"] == "3"
    run = runs.find_last(r=2)
    assert isinstance(run, Run)
    assert run.data.params["p"] == "5"


def test_runs_find_last_none(runs: Runs):
    run = runs.find_last({"p": 10})
    assert run is None
