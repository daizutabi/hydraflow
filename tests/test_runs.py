from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import mlflow
import pytest
from hydraflow.runs import Runs
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


def test_filter_invalid_param(runs: list[Run]):
    from hydraflow.runs import filter_runs

    x = filter_runs(runs, {"invalid": 0})
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


@pytest.fixture
def mock_runs():
    from hydraflow.runs import Runs

    run1 = MagicMock()
    run1.info.start_time = 1
    run1.data.params = {"param1": "value1", "param2": "value2"}

    run2 = MagicMock()
    run2.info.start_time = 2
    run2.data.params = {"param1": "value1", "param2": "value3"}

    run3 = MagicMock()
    run3.info.start_time = 3
    run3.data.params = {"param1": "value4", "param2": "value2"}

    return Runs([run1, run2, run3])


def test_filter_runs(mock_runs: Runs):
    filtered_runs = mock_runs.filter({"param1": "value1"})
    assert len(filtered_runs) == 2


def test_get_run_by_params(mock_runs: Runs):
    run = mock_runs.get({"param1": "value4"})
    assert run is not None
    assert run.data.params["param1"] == "value4"


def test_get_earliest_run(mock_runs: Runs):
    earliest_run = mock_runs.get_earliest_run()
    assert earliest_run
    assert earliest_run.info.start_time == 1


def test_get_latest_run(mock_runs: Runs):
    latest_run = mock_runs.get_latest_run()
    assert latest_run
    assert latest_run.info.start_time == 3


def test_get_param_names_mock(mock_runs: Runs):
    param_names = mock_runs.get_param_names()
    assert set(param_names) == {"param1", "param2"}


def test_get_param_dict_mock(mock_runs: Runs):
    param_dict = mock_runs.get_param_dict()
    assert param_dict == {"param1": ["value1", "value4"], "param2": ["value2", "value3"]}


def test_filter_runs_no_match(runs: list[Run]):
    from hydraflow.runs import filter_runs

    assert len(runs) == 6
    x = filter_runs(runs, {"p": 999})
    assert len(x) == 0


def test_get_run_no_match(runs: list[Run]):
    from hydraflow.runs import get_run

    run = get_run(runs, {"p": 999})
    assert run is None


def test_get_run_multiple_matches(runs: list[Run]):
    from hydraflow.runs import get_run

    with pytest.raises(ValueError):
        get_run(runs, {"q": 0})


def test_get_earliest_run_no_match(runs: list[Run]):
    from hydraflow.runs import get_earliest_run

    run = get_earliest_run(runs, {"p": 999})
    assert run is None


def test_get_latest_run_no_match(runs: list[Run]):
    from hydraflow.runs import get_latest_run

    run = get_latest_run(runs, {"p": 999})
    assert run is None


@pytest.fixture
def mock_mlflow_search_runs():
    with patch("mlflow.search_runs") as mock_search_runs:
        yield mock_search_runs


def test_search_runs(mock_mlflow_search_runs):
    from hydraflow.runs import search_runs

    mock_run = MagicMock()
    mock_mlflow_search_runs.return_value = [mock_run]

    result = search_runs(
        experiment_ids=["1"],
        filter_string="metrics.accuracy > 0.9",
        run_view_type=1,
        max_results=10,
        order_by=["metrics.accuracy DESC"],
        search_all_experiments=False,
        experiment_names=None,
    )

    assert isinstance(result, Runs)
    assert len(result.runs) == 1
    assert result.runs[0] == mock_run


def test_search_runs_no_results(mock_mlflow_search_runs):
    from hydraflow.runs import search_runs

    mock_mlflow_search_runs.return_value = []

    result = search_runs(
        experiment_ids=["1"],
        filter_string="metrics.accuracy > 0.9",
        run_view_type=1,
        max_results=10,
        order_by=["metrics.accuracy DESC"],
        search_all_experiments=False,
        experiment_names=None,
    )

    assert isinstance(result, Runs)
    assert len(result.runs) == 0


def test_load_config_empty(runs: list[Run]):
    from hydraflow.runs import load_config

    assert load_config(runs[0]) == {}
