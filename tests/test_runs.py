from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from mlflow.entities import Run
from omegaconf import DictConfig

from hydraflow.runs import RunCollection


@pytest.fixture
def runs(monkeypatch, tmp_path):
    from hydraflow.runs import search_runs

    monkeypatch.chdir(tmp_path)

    mlflow.set_experiment("test_run")
    for x in range(6):
        with mlflow.start_run(run_name=f"{x}"):
            mlflow.log_param("p", x)
            mlflow.log_param("q", 0 if x < 5 else None)
            mlflow.log_param("r", x % 3)
            mlflow.log_text(f"{x}", "abc.txt")

    x = search_runs()
    assert isinstance(x, RunCollection)
    return x


@pytest.fixture
def run_list(runs: RunCollection):
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
    assert len(x) == 5
    x = filter_runs(run_list, q=0)
    assert len(x) == 5


def test_filter_list(run_list: list[Run]):
    from hydraflow.runs import filter_runs

    x = filter_runs(run_list, p=[0, 4, 5])
    assert len(x) == 3


def test_filter_tuple(run_list: list[Run]):
    from hydraflow.runs import filter_runs

    x = filter_runs(run_list, p=(1, 3))
    assert len(x) == 2


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
    assert len(params["q"]) == 2


@pytest.mark.parametrize("i", range(6))
def test_chdir_artifact_list(i: int, run_list: list[Run]):
    from hydraflow.context import chdir_artifact

    with chdir_artifact(run_list[i]):
        assert Path("abc.txt").read_text() == f"{i}"

    assert not Path("abc.txt").exists()


def test_runs_repr(runs: RunCollection):
    assert repr(runs) == "RunCollection(6)"


def test_runs_first(runs: RunCollection):
    run = runs.first()
    assert isinstance(run, Run)
    assert run.data.params["p"] == "0"


def test_runs_last(runs: RunCollection):
    run = runs.last()
    assert isinstance(run, Run)
    assert run.data.params["p"] == "5"


def test_runs_filter(runs: RunCollection):
    assert len(runs.filter()) == 6
    assert len(runs.filter({})) == 6
    assert len(runs.filter({"p": 1})) == 1
    assert len(runs.filter({"q": 0})) == 5
    assert len(runs.filter({"q": -1})) == 0
    assert len(runs.filter(p=5)) == 1
    assert len(runs.filter(q=0)) == 5
    assert len(runs.filter(q=-1)) == 0
    assert len(runs.filter({"r": 2})) == 2
    assert len(runs.filter(r=0)) == 2


def test_runs_get(runs: RunCollection):
    from hydraflow.runs import Run

    run = runs.get({"p": 4})
    assert isinstance(run, Run)
    run = runs.get(p=2)
    assert isinstance(run, Run)


def test_runs_try_get(runs: RunCollection):
    run = runs.try_get({"p": 5})
    assert isinstance(run, Run)
    run = runs.try_get(p=1)
    assert isinstance(run, Run)
    run = runs.try_get(p=-1)
    assert run is None


def test_runs_get_params_names(runs: RunCollection):
    names = runs.get_param_names()
    assert len(names) == 3
    assert "p" in names
    assert "q" in names
    assert "r" in names


def test_runs_get_params_dict(runs: RunCollection):
    params = runs.get_param_dict()
    assert params["p"] == ["0", "1", "2", "3", "4", "5"]
    assert params["q"] == ["0", "None"]
    assert params["r"] == ["0", "1", "2"]


def test_runs_find(runs: RunCollection):
    from hydraflow.runs import Run

    run = runs.find({"r": 0})
    assert isinstance(run, Run)
    assert run.data.params["p"] == "0"
    run = runs.find(r=2)
    assert isinstance(run, Run)
    assert run.data.params["p"] == "2"


def test_runs_find_none(runs: RunCollection):
    with pytest.raises(ValueError):
        runs.find({"r": 10})


def test_runs_try_find_none(runs: RunCollection):
    run = runs.try_find({"r": 10})
    assert run is None


def test_runs_find_last(runs: RunCollection):
    from hydraflow.runs import Run

    run = runs.find_last({"r": 0})
    assert isinstance(run, Run)
    assert run.data.params["p"] == "3"
    run = runs.find_last(r=2)
    assert isinstance(run, Run)
    assert run.data.params["p"] == "5"


def test_runs_find_last_none(runs: RunCollection):
    with pytest.raises(ValueError):
        runs.find_last({"p": 10})


def test_runs_try_find_last_none(runs: RunCollection):
    run = runs.try_find_last({"p": 10})
    assert run is None


@pytest.fixture
def runs2(monkeypatch, tmp_path):
    mlflow.set_experiment("test_run2")
    for x in range(3):
        with mlflow.start_run(run_name=f"{x}"):
            mlflow.log_param("x", x)


def test_list_runs(runs, runs2):
    from hydraflow.runs import list_runs

    mlflow.set_experiment("test_run")
    all_runs = list_runs()
    assert len(all_runs) == 6

    mlflow.set_experiment("test_run2")
    all_runs = list_runs()
    assert len(all_runs) == 3


def test_list_runs_empty_list(runs, runs2):
    from hydraflow.runs import list_runs

    all_runs = list_runs([])
    assert len(all_runs) == 9


@pytest.mark.parametrize(["name", "n"], [("test_run", 6), ("test_run2", 3)])
def test_list_runs_list(runs, runs2, name, n):
    from hydraflow.runs import list_runs

    filtered_runs = list_runs(experiment_names=[name])
    assert len(filtered_runs) == n


def test_list_runs_none(runs, runs2):
    from hydraflow.runs import list_runs

    no_runs = list_runs(experiment_names=["non_existent_experiment"])
    assert len(no_runs) == 0


def test_run_collection_map(runs: RunCollection):
    results = list(runs.map(lambda run: run.info.run_id))
    assert len(results) == len(runs._runs)
    assert all(isinstance(run_id, str) for run_id in results)


def test_run_collection_map_run_id(runs: RunCollection):
    results = list(runs.map_run_id(lambda run_id: run_id))
    assert len(results) == len(runs._runs)
    assert all(isinstance(run_id, str) for run_id in results)


def test_run_collection_map_config(runs: RunCollection):
    results = list(runs.map_config(lambda config: config))
    assert len(results) == len(runs._runs)
    assert all(isinstance(config, DictConfig) for config in results)


def test_run_collection_map_uri(runs: RunCollection):
    results = list(runs.map_uri(lambda uri: uri))
    assert len(results) == len(runs._runs)
    assert all(isinstance(uri, (str, type(None))) for uri in results)


def test_run_collection_map_dir(runs: RunCollection):
    results = list(runs.map_dir(lambda dir_path: dir_path))
    assert len(results) == len(runs._runs)
    assert all(isinstance(dir_path, str) for dir_path in results)


# def test_hydra_output_dir_error(runs_list: list[Run]):
#     from hydraflow.runs import get_hydra_output_dir

#     with pytest.raises(FileNotFoundError):
#         get_hydra_output_dir(runs_list[0])
