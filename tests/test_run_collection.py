from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from mlflow.entities import Run, RunStatus

from hydraflow.run_collection import RunCollection


@pytest.fixture
def rc(monkeypatch, tmp_path):
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


def test_bool_false():
    assert not RunCollection([])
    assert bool(RunCollection.from_list([])) is False


def test_bool_true(rc: RunCollection):
    assert rc
    assert bool(rc) is True


@pytest.fixture
def run_list(rc: RunCollection):
    return rc._runs


def test_from_list(run_list: list[Run]):
    rc = RunCollection.from_list(run_list)
    assert len(rc) == len(run_list)
    assert all(run in rc for run in run_list)


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
    x = filter_runs(run_list, ["p=1"])
    assert len(x) == 1


def test_filter_all(run_list: list[Run]):
    from hydraflow.run_collection import filter_runs

    assert len(run_list) == 6
    x = filter_runs(run_list, {"q": 0})
    assert len(x) == 5
    x = filter_runs(run_list, q=0)
    assert len(x) == 5
    x = filter_runs(run_list, ["q=0"])
    assert len(x) == 5


def test_filter_list(run_list: list[Run]):
    from hydraflow.run_collection import filter_runs

    x = filter_runs(run_list, p=[0, 4, 5])
    assert len(x) == 3


def test_filter_tuple(run_list: list[Run]):
    from hydraflow.run_collection import filter_runs

    x = filter_runs(run_list, p=(1, 3))
    assert len(x) == 3


def test_filter_invalid_param(run_list: list[Run]):
    from hydraflow.run_collection import filter_runs

    x = filter_runs(run_list, {"invalid": 0})
    assert len(x) == 0
    x = filter_runs(run_list, ["invalid=0"])
    assert len(x) == 0


def test_filter_status(run_list: list[Run]):
    from hydraflow.run_collection import filter_runs

    assert not filter_runs(run_list, status="RUNNING")
    assert filter_runs(run_list, status="finished") == run_list
    assert filter_runs(run_list, status=["finished", "running"]) == run_list
    assert filter_runs(run_list, status="!RUNNING") == run_list
    assert not filter_runs(run_list, status="!finished")


def test_filter_status_enum(run_list: list[Run]):
    from hydraflow.run_collection import filter_runs

    assert not filter_runs(run_list, status=RunStatus.RUNNING)
    assert filter_runs(run_list, status=RunStatus.FINISHED) == run_list
    s = [RunStatus.FINISHED, RunStatus.RUNNING]
    assert filter_runs(run_list, status=s) == run_list


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


def test_repr(rc: RunCollection):
    assert repr(rc) == "RunCollection(6)"


def test_first(rc: RunCollection):
    run = rc.first()
    assert isinstance(run, Run)
    assert run.data.params["p"] == "0"


def test_first_empty(rc: RunCollection):
    rc._runs = []
    with pytest.raises(ValueError):
        rc.first()


def test_try_first_none(rc: RunCollection):
    rc._runs = []
    assert rc.try_first() is None


def test_last(rc: RunCollection):
    run = rc.last()
    assert isinstance(run, Run)
    assert run.data.params["p"] == "5"


def test_last_empty(rc: RunCollection):
    rc._runs = []
    with pytest.raises(ValueError):
        rc.last()


def test_try_last_none(rc: RunCollection):
    rc._runs = []
    assert rc.try_last() is None


def test_filter(rc: RunCollection):
    assert len(rc.filter()) == 6
    assert len(rc.filter({})) == 6
    assert len(rc.filter({"p": 1})) == 1
    assert len(rc.filter(["p=1"])) == 1
    assert len(rc.filter({"q": 0})) == 5
    assert len(rc.filter(["q=0"])) == 5
    assert len(rc.filter({"q": -1})) == 0
    assert len(rc.filter(["q=-1"])) == 0
    assert not rc.filter({"q": -1})
    assert len(rc.filter(p=5)) == 1
    assert len(rc.filter(q=0)) == 5
    assert len(rc.filter(q=-1)) == 0
    assert not rc.filter(q=-1)
    assert len(rc.filter({"r": 2})) == 2
    assert len(rc.filter(["r=2"])) == 2
    assert len(rc.filter(r=0)) == 2
    assert len(rc.filter(["r=0"])) == 2


def test_get(rc: RunCollection):
    run = rc.get({"p": 4})
    assert isinstance(run, Run)
    run = rc.get(p=2)
    assert isinstance(run, Run)
    run = rc.get(["p=3"])
    assert isinstance(run, Run)


def test_try_get(rc: RunCollection):
    run = rc.try_get({"p": 5})
    assert isinstance(run, Run)
    run = rc.try_get(["p=2"])
    assert isinstance(run, Run)
    run = rc.try_get(p=1)
    assert isinstance(run, Run)
    run = rc.try_get(p=-1)
    assert run is None
    run = rc.try_get(["p=-2"])
    assert run is None


def test_get_param_names(rc: RunCollection):
    names = rc.get_param_names()
    assert len(names) == 3
    assert "p" in names
    assert "q" in names
    assert "r" in names


def test_get_param_dict(rc: RunCollection):
    params = rc.get_param_dict()
    assert params["p"] == ["0", "1", "2", "3", "4", "5"]
    assert params["q"] == ["0", "None"]
    assert params["r"] == ["0", "1", "2"]


def test_get_param_dict_drop_const(rc: RunCollection):
    rc_ = rc.filter(q=0)
    params = rc_.get_param_dict(drop_const=True)
    assert len(params) == 2
    assert "p" in params
    assert "q" not in params
    assert "r" in params


def test_find(rc: RunCollection):
    run = rc.find({"r": 0})
    assert isinstance(run, Run)
    assert run.data.params["p"] == "0"
    run = rc.find(r=2)
    assert isinstance(run, Run)
    assert run.data.params["p"] == "2"


def test_find_none(rc: RunCollection):
    with pytest.raises(ValueError):
        rc.find({"r": 10})


def test_try_find_none(rc: RunCollection):
    run = rc.try_find({"r": 10})
    assert run is None


def test_find_last(rc: RunCollection):
    run = rc.find_last({"r": 0})
    assert isinstance(run, Run)
    assert run.data.params["p"] == "3"
    run = rc.find_last(r=2)
    assert isinstance(run, Run)
    assert run.data.params["p"] == "5"


def test_find_last_none(rc: RunCollection):
    with pytest.raises(ValueError):
        rc.find_last({"p": 10})


def test_try_find_last_none(rc: RunCollection):
    run = rc.try_find_last({"p": 10})
    assert run is None


@pytest.fixture
def runs2(monkeypatch, tmp_path):
    mlflow.set_experiment("test_run2")
    for x in range(3):
        with mlflow.start_run(run_name=f"{x}"):
            mlflow.log_param("x", x)


def test_list_runs(rc, runs2):
    from hydraflow.mlflow import list_runs

    mlflow.set_experiment("test_run")
    all_runs = list_runs()
    assert len(all_runs) == 6

    mlflow.set_experiment("test_run2")
    all_runs = list_runs()
    assert len(all_runs) == 3


def test_list_runs_empty_list(rc, runs2):
    from hydraflow.mlflow import list_runs

    all_runs = list_runs([])
    assert len(all_runs) == 9


@pytest.mark.parametrize(["name", "n"], [("test_run", 6), ("test_run2", 3)])
def test_list_runs_list(rc, runs2, name, n):
    from hydraflow.mlflow import list_runs

    filtered_runs = list_runs(name)
    assert len(filtered_runs) == n


def test_list_runs_none(rc, runs2):
    from hydraflow.mlflow import list_runs

    no_runs = list_runs(["non_existent_experiment"])
    assert len(no_runs) == 0
    assert not no_runs


def test_map(rc: RunCollection):
    results = list(rc.map(lambda run: run.info.run_id))
    assert len(results) == len(rc._runs)
    assert all(isinstance(run_id, str) for run_id in results)


def test_map_args(rc: RunCollection):
    results = list(rc.map(lambda run, x: run.info.run_id + x, "test"))
    assert all(x.endswith("test") for x in results)


def test_map_id(rc: RunCollection):
    results = list(rc.map_id(lambda run_id: run_id))
    assert len(results) == len(rc._runs)
    assert all(isinstance(run_id, str) for run_id in results)


def test_map_id_kwargs(rc: RunCollection):
    results = list(rc.map_id(lambda run_id, x: x + run_id, x="test"))
    assert all(x.startswith("test") for x in results)


def test_map_uri(rc: RunCollection):
    results = list(rc.map_uri(lambda uri: uri))
    assert len(results) == len(rc._runs)
    assert all(isinstance(uri, str | type(None)) for uri in results)


def test_map_dir(rc: RunCollection):
    results = list(rc.map_dir(lambda dir_path, x: dir_path / x, "a.csv"))
    assert len(results) == len(rc._runs)
    assert all(isinstance(dir_path, Path) for dir_path in results)
    assert all(dir_path.stem == "a" for dir_path in results)


def test_sort(rc: RunCollection):
    rc.sort(key=lambda x: x.data.params["p"])
    assert [run.data.params["p"] for run in rc] == ["0", "1", "2", "3", "4", "5"]

    rc.sort(reverse=True)
    assert [run.data.params["p"] for run in rc] == ["5", "4", "3", "2", "1", "0"]


def test_iter(rc: RunCollection):
    assert list(rc) == rc._runs


@pytest.mark.parametrize("i", range(6))
def test_run_collection_getitem(rc: RunCollection, i: int):
    assert rc[i] == rc._runs[i]


@pytest.mark.parametrize("i", range(6))
def test_getitem_slice(rc: RunCollection, i: int):
    assert rc[i : i + 2]._runs == rc._runs[i : i + 2]


@pytest.mark.parametrize("i", range(6))
def test_getitem_slice_step(rc: RunCollection, i: int):
    assert rc[i::2]._runs == rc._runs[i::2]


@pytest.mark.parametrize("i", range(6))
def test_getitem_slice_step_neg(rc: RunCollection, i: int):
    assert rc[i::-2]._runs == rc._runs[i::-2]


def test_take(rc: RunCollection):
    assert rc.take(3)._runs == rc._runs[:3]
    assert len(rc.take(4)) == 4
    assert rc.take(10)._runs == rc._runs


def test_take_neg(rc: RunCollection):
    assert rc.take(-3)._runs == rc._runs[-3:]
    assert len(rc.take(-4)) == 4
    assert rc.take(-10)._runs == rc._runs


@pytest.mark.parametrize("i", range(6))
def test_contains(rc: RunCollection, i: int):
    assert rc[i] in rc
    assert rc._runs[i] in rc


def test_group_by(rc: RunCollection):
    grouped = rc.group_by(["p"])
    assert len(grouped) == 6
    assert all(isinstance(group, RunCollection) for group in grouped.values())
    assert all(len(group) == 1 for group in grouped.values())
    assert grouped[("0",)][0] == rc[0]
    assert grouped[("1",)][0] == rc[1]

    grouped = rc.group_by("q")
    assert len(grouped) == 2

    grouped = rc.group_by("r")
    assert len(grouped) == 3


def test_filter_runs_empty_list():
    from hydraflow.run_collection import filter_runs

    x = filter_runs([], p=[0, 1, 2])
    assert x == []


def test_filter_runs_no_match(run_list: list[Run]):
    from hydraflow.run_collection import filter_runs

    x = filter_runs(run_list, p=[10, 11, 12])
    assert x == []


def test_get_run_no_match(rc: RunCollection):
    with pytest.raises(ValueError):
        rc.get({"p": 10})


def test_get_run_multiple_params(rc: RunCollection):
    run = rc.get({"p": 4, "q": 0})
    assert isinstance(run, Run)
    assert run.data.params["p"] == "4"
    assert run.data.params["q"] == "0"


def test_try_get_run_no_match(rc: RunCollection):
    assert rc.try_get({"p": 10}) is None


def test_try_get_run_multiple_params(rc: RunCollection):
    run = rc.try_get({"p": 4, "q": 0})
    assert isinstance(run, Run)
    assert run.data.params["p"] == "4"
    assert run.data.params["q"] == "0"
