from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
from mlflow.entities import Run
from pandas import DataFrame, Series


@pytest.fixture
def _runs(monkeypatch, tmp_path):
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
    y = mlflow.search_runs(output_format="pandas", order_by=["params.p"])
    assert isinstance(y, DataFrame)
    return x, y


@pytest.fixture(params=["list", "pandas"])
def runs(_runs: tuple[list[Run], DataFrame], request: pytest.FixtureRequest):
    if request.param == "list":
        return _runs[0]

    return _runs[1]


@pytest.fixture
def runs_list(_runs: tuple[list[Run], DataFrame], request: pytest.FixtureRequest):
    return _runs[0]


@pytest.fixture
def runs_df(_runs: tuple[list[Run], DataFrame], request: pytest.FixtureRequest):
    return _runs[1]


def test_filter_by_config_one(runs: list[Run] | DataFrame):
    from hydraflow.run import filter_by_config

    assert len(runs) == 6
    x = filter_by_config(runs, {"p": 1})
    assert len(x) == 1


def test_filter_by_config_all(runs: list[Run] | DataFrame):
    from hydraflow.run import filter_by_config

    assert len(runs) == 6
    x = filter_by_config(runs, {"q": 0})
    assert len(x) == 6


def test_get_by_config_list(runs_list: list[Run]):
    from hydraflow.run import get_by_config

    run = get_by_config(runs_list, {"p": 4})
    assert isinstance(run, Run)
    assert run.data.params["p"] == "4"


def test_get_by_config_df(runs_df: DataFrame):
    from hydraflow.run import get_by_config

    run = get_by_config(runs_df, {"p": 2})
    assert isinstance(run, Series)
    assert run["params.p"] == "2"


def test_get_by_config_error(runs: list[Run] | DataFrame):
    from hydraflow.run import get_by_config

    with pytest.raises(ValueError):
        get_by_config(runs, {"q": 0})


def test_drop_unique_params(runs_df):
    from hydraflow.run import drop_unique_params

    assert "params.p" in runs_df
    assert "params.q" in runs_df
    df = drop_unique_params(runs_df)
    assert "params.p" in df
    assert "params.q" not in df


def test_get_param_names(runs_df: DataFrame):
    from hydraflow.run import get_param_names

    params = get_param_names(runs_df)
    assert len(params) == 2
    assert "p" in params
    assert "q" in params


def test_get_param_dict(runs_df: DataFrame):
    from hydraflow.run import get_param_dict

    params = get_param_dict(runs_df)
    assert len(params["p"]) == 6
    assert len(params["q"]) == 1


@pytest.mark.parametrize("i", range(6))
def test_get_run_id(i: int, runs_list: list[Run], runs_df: DataFrame):
    from hydraflow.run import get_run_id

    assert get_run_id(runs_list[i]) == get_run_id(runs_df.iloc[i])
    assert get_run_id(runs_list[i]) == get_run_id(runs_df.iloc[i])

    x = get_run_id(runs_list[i])
    assert get_run_id(x) == runs_list[i].info.run_id


@pytest.mark.parametrize("i", range(6))
@pytest.mark.parametrize("path", [None, "a"])
def test_get_artifact_uri(i: int, path, runs_list: list[Run], runs_df: DataFrame):
    from hydraflow.run import get_artifact_uri, get_run_id

    x = get_run_id(runs_list[i])
    y = get_artifact_uri(runs_list[i], path)
    assert get_artifact_uri(x, path) == y
    assert get_artifact_uri(runs_df.iloc[i], path) == y


@pytest.mark.parametrize("i", range(6))
def test_chdir_artifact_list(i: int, runs_list: list[Run]):
    from hydraflow.context import chdir_artifact

    with chdir_artifact(runs_list[i]):
        assert Path("abc.txt").read_text() == f"{i}"

    assert not Path("abc.txt").exists()


def test_hydra_output_dir_error(runs_list: list[Run]):
    from hydraflow.run import get_hydra_output_dir

    with pytest.raises(FileNotFoundError):
        get_hydra_output_dir(runs_list[0])


@pytest.fixture
def df():
    return DataFrame(
        {
            "a": [0, 0, 0, 0],
            "params.x": [1, 1, 2, 2],
            "params.y": [1, 2, 1, 2],
            "params.z": [1, 1, 1, 1],
        },
    )


def test_unique_params(df):
    from hydraflow.run import drop_unique_params

    df = drop_unique_params(df)
    assert len(df.columns) == 3
    assert "a" in df
    assert "params.x" in df
    assert "params.z" not in df


def test_param_names(df):
    from hydraflow.run import get_param_names

    names = get_param_names(df)
    assert names == ["x", "y", "z"]


def test_param_dict(df):
    from hydraflow.run import get_param_dict

    x = get_param_dict(df)
    assert x["x"] == [1, 2]
    assert x["y"] == [1, 2]
    assert x["z"] == [1]
