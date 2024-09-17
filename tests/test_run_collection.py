from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from mlflow.entities import Run

from hydraflow.run_collection import RunCollection


@pytest.fixture
def runs(monkeypatch, tmp_path):
    from hydraflow.mlflow import search_runs

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
    from hydraflow.run_collection import filter_runs

    assert run_list == filter_runs(run_list)


def test_filter_one(run_list: list[Run]):
    from hydraflow.run_collection import filter_runs

    assert len(run_list) == 6
    x = filter_runs(run_list, {"p": 1})
    assert len(x) == 1
    x = filter_runs(run_list, p=1)
    assert len(x) == 1


def test_filter_all(run_list: list[Run]):
    from hydraflow.run_collection import filter_runs

    assert len(run_list) == 6
    x = filter_runs(run_list, {"q": 0})
    assert len(x) == 5
    x = filter_runs(run_list, q=0)
    assert len(x) == 5


def test_filter_list(run_list: list[Run]):
    from hydraflow.run_collection import filter_runs

    x = filter_runs(run_list, p=[0, 4, 5])
    assert len(x) == 3


def test_filter_tuple(run_list: list[Run]):
    from hydraflow.run_collection import filter_runs

    x = filter_runs(run_list, p=(1, 3))
    assert len(x) == 2


def test_filter_invalid_param(run_list: list[Run]):
    from hydraflow.run_collection import filter_runs

    x = filter_runs(run_list, {"invalid": 0})
    assert len(x) == 6


def test_get_params(run_list: list[Run]):
    from hydraflow.run_collection import get_params

    assert get_params(run_list[1], "p") == ("1",)
    assert get_params(run_list[2], "p", "q") == ("2", "0")
    assert get_params(run_list[3], ["p", "q"]) == ("3", "0")
    assert get_params(run_list[4], "p", ["q", "r"]) == ("4", "0", "1")
    assert get_params(run_list[5], ["a", "q"], "r") == (None, "None", "2")


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


def test_runs_first_empty(runs: RunCollection):
    runs._runs = []
    with pytest.raises(ValueError):
        runs.first()


def test_runs_try_first_none(runs: RunCollection):
    runs._runs = []
    assert runs.try_first() is None


def test_runs_last(runs: RunCollection):
    run = runs.last()
    assert isinstance(run, Run)
    assert run.data.params["p"] == "5"


def test_runs_last_empty(runs: RunCollection):
    runs._runs = []
    with pytest.raises(ValueError):
        runs.last()


def test_runs_try_last_none(runs: RunCollection):
    runs._runs = []
    assert runs.try_last() is None


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
    from hydraflow.run_collection import Run

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
    from hydraflow.run_collection import Run

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
    from hydraflow.run_collection import Run

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
    from hydraflow.mlflow import list_runs

    mlflow.set_experiment("test_run")
    all_runs = list_runs()
    assert len(all_runs) == 6

    mlflow.set_experiment("test_run2")
    all_runs = list_runs()
    assert len(all_runs) == 3


def test_list_runs_empty_list(runs, runs2):
    from hydraflow.mlflow import list_runs

    all_runs = list_runs([])
    assert len(all_runs) == 9


@pytest.mark.parametrize(["name", "n"], [("test_run", 6), ("test_run2", 3)])
def test_list_runs_list(runs, runs2, name, n):
    from hydraflow.mlflow import list_runs

    filtered_runs = list_runs(name)
    assert len(filtered_runs) == n


def test_list_runs_none(runs, runs2):
    from hydraflow.mlflow import list_runs

    no_runs = list_runs(["non_existent_experiment"])
    assert len(no_runs) == 0


def test_run_collection_map(runs: RunCollection):
    results = list(runs.map(lambda run: run.info.run_id))
    assert len(results) == len(runs._runs)
    assert all(isinstance(run_id, str) for run_id in results)


def test_run_collection_map_args(runs: RunCollection):
    results = list(runs.map(lambda run, x: run.info.run_id + x, "test"))
    assert all(x.endswith("test") for x in results)


def test_run_collection_map_run_id(runs: RunCollection):
    results = list(runs.map_run_id(lambda run_id: run_id))
    assert len(results) == len(runs._runs)
    assert all(isinstance(run_id, str) for run_id in results)


def test_run_collection_map_run_id_kwargs(runs: RunCollection):
    results = list(runs.map_run_id(lambda run_id, x: x + run_id, x="test"))
    assert all(x.startswith("test") for x in results)


def test_run_collection_map_uri(runs: RunCollection):
    results = list(runs.map_uri(lambda uri: uri))
    assert len(results) == len(runs._runs)
    assert all(isinstance(uri, (str, type(None))) for uri in results)


def test_run_collection_map_dir(runs: RunCollection):
    results = list(runs.map_dir(lambda dir_path, x: dir_path / x, "a.csv"))
    assert len(results) == len(runs._runs)
    assert all(isinstance(dir_path, Path) for dir_path in results)
    assert all(dir_path.stem == "a" for dir_path in results)


def test_run_collection_sort(runs: RunCollection):
    runs.sort(key=lambda x: x.data.params["p"])
    assert [run.data.params["p"] for run in runs] == ["0", "1", "2", "3", "4", "5"]

    runs.sort(reverse=True)
    assert [run.data.params["p"] for run in runs] == ["5", "4", "3", "2", "1", "0"]


def test_run_collection_iter(runs: RunCollection):
    assert list(runs) == runs._runs


@pytest.mark.parametrize("i", range(6))
def test_run_collection_getitem(runs: RunCollection, i: int):
    assert runs[i] == runs._runs[i]


@pytest.mark.parametrize("i", range(6))
def test_run_collection_getitem_slice(runs: RunCollection, i: int):
    assert runs[i : i + 2]._runs == runs._runs[i : i + 2]


@pytest.mark.parametrize("i", range(6))
def test_run_collection_getitem_slice_step(runs: RunCollection, i: int):
    assert runs[i::2]._runs == runs._runs[i::2]


@pytest.mark.parametrize("i", range(6))
def test_run_collection_getitem_slice_step_neg(runs: RunCollection, i: int):
    assert runs[i::-2]._runs == runs._runs[i::-2]


def test_run_collection_take(runs: RunCollection):
    assert runs.take(3)._runs == runs._runs[:3]
    assert len(runs.take(4)) == 4
    assert runs.take(10)._runs == runs._runs


def test_run_collection_take_neg(runs: RunCollection):
    assert runs.take(-3)._runs == runs._runs[-3:]
    assert len(runs.take(-4)) == 4
    assert runs.take(-10)._runs == runs._runs


@pytest.mark.parametrize("i", range(6))
def test_run_collection_contains(runs: RunCollection, i: int):
    assert runs[i] in runs
    assert runs._runs[i] in runs


def test_run_collection_group_by(runs: RunCollection):
    grouped = runs.group_by(["p"])
    assert len(grouped) == 6
    assert all(isinstance(group, RunCollection) for group in grouped.values())
    assert all(len(group) == 1 for group in grouped.values())
    assert grouped[("0",)][0] == runs[0]
    assert grouped[("1",)][0] == runs[1]

    grouped = runs.group_by("q")
    assert len(grouped) == 2

    grouped = runs.group_by("r")
    assert len(grouped) == 3


def test_filter_runs_empty_list():
    from hydraflow.run_collection import filter_runs

    x = filter_runs([], p=[0, 1, 2])
    assert x == []


def test_filter_runs_no_match(run_list: list[Run]):
    from hydraflow.run_collection import filter_runs

    x = filter_runs(run_list, p=[10, 11, 12])
    assert x == []


def test_get_run_no_match(runs: RunCollection):
    with pytest.raises(ValueError):
        runs.get({"p": 10})


def test_get_run_multiple_params(runs: RunCollection):
    run = runs.get({"p": 4, "q": 0})
    assert isinstance(run, Run)
    assert run.data.params["p"] == "4"
    assert run.data.params["q"] == "0"


def test_try_get_run_no_match(runs: RunCollection):
    assert runs.try_get({"p": 10}) is None


def test_try_get_run_multiple_params(runs: RunCollection):
    run = runs.try_get({"p": 4, "q": 0})
    assert isinstance(run, Run)
    assert run.data.params["p"] == "4"
    assert run.data.params["q"] == "0"
