from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
from mlflow.entities.run import Run as Run_
from mlflow.tracking import artifact_utils
from omegaconf import DictConfig, OmegaConf
from pandas import DataFrame, Series

from hydraflow.config import iter_params
from hydraflow.util import uri_to_path

if TYPE_CHECKING:
    from typing import Any


@dataclass
class Runs:
    runs: list[Run_] | DataFrame

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)})"

    def __len__(self) -> int:
        return len(self.runs)

    def filter(self, config: object) -> Runs:
        return Runs(filter_runs(self.runs, config))

    def get(self, config: object) -> Run:
        return Run(get_run(self.runs, config))

    def drop_unique_params(self) -> Runs:
        if isinstance(self.runs, DataFrame):
            return Runs(drop_unique_params(self.runs))

        raise NotImplementedError

    def get_param_names(self) -> list[str]:
        if isinstance(self.runs, DataFrame):
            return get_param_names(self.runs)

        raise NotImplementedError

    def get_param_dict(self) -> dict[str, list[str]]:
        if isinstance(self.runs, DataFrame):
            return get_param_dict(self.runs)

        raise NotImplementedError


def filter_runs(runs: list[Run_] | DataFrame, config: object) -> list[Run_] | DataFrame:
    if isinstance(runs, list):
        return filter_runs_list(runs, config)

    return filter_runs_dataframe(runs, config)


def _is_equal(run: Run_, key: str, value: Any) -> bool:
    param = run.data.params.get(key, value)

    if param is None:
        return False

    return type(value)(param) == value


def filter_runs_list(runs: list[Run_], config: object) -> list[Run_]:
    for key, value in iter_params(config):
        runs = [run for run in runs if _is_equal(run, key, value)]

    return runs


def filter_runs_dataframe(runs: DataFrame, config: object) -> DataFrame:
    index = np.ones(len(runs), dtype=bool)

    for key, value in iter_params(config):
        name = f"params.{key}"

        if name in runs:
            series = runs[name]
            is_value = -series.isna()
            param = series.fillna(value).astype(type(value))
            index &= is_value & (param == value)

    return runs[index]


def get_run(runs: list[Run_] | DataFrame, config: object) -> Run_ | Series:
    runs = filter_runs(runs, config)

    if len(runs) == 1:
        return runs[0] if isinstance(runs, list) else runs.iloc[0]

    msg = f"number of filtered runs is not 1: got {len(runs)}"
    raise ValueError(msg)


def drop_unique_params(runs: DataFrame) -> DataFrame:
    def select(column: str) -> bool:
        return not column.startswith("params.") or len(runs[column].unique()) > 1

    columns = [select(column) for column in runs.columns]
    return runs.iloc[:, columns]


def get_param_names(runs: DataFrame) -> list[str]:
    def get_name(column: str) -> str:
        if column.startswith("params."):
            return column.split(".", maxsplit=1)[-1]

        return ""

    columns = [get_name(column) for column in runs.columns]
    return [column for column in columns if column]


def get_param_dict(runs: DataFrame) -> dict[str, list[str]]:
    params = {}
    for name in get_param_names(runs):
        params[name] = list(runs[f"params.{name}"].unique())

    return params


@dataclass
class Run:
    run: Run_ | Series | str

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.run_id!r})"

    @property
    def run_id(self) -> str:
        return get_run_id(self.run)

    def artifact_uri(self, artifact_path: str | None = None) -> str:
        return get_artifact_uri(self.run, artifact_path)

    @property
    def artifact_dir(self) -> Path:
        return get_artifact_dir(self.run)

    def artifact_path(self, artifact_path: str | None = None) -> Path:
        return get_artifact_path(self.run, artifact_path)

    @property
    def config(self) -> DictConfig:
        return load_config(self.run)

    def log_hydra_output_dir(self) -> None:
        log_hydra_output_dir(self.run)


def get_run_id(run: Run_ | Series | str) -> str:
    if isinstance(run, str):
        return run

    if isinstance(run, Run_):
        return run.info.run_id

    return run.run_id


def get_artifact_uri(run: Run_ | Series | str, artifact_path: str | None = None) -> str:
    run_id = get_run_id(run)
    return artifact_utils.get_artifact_uri(run_id, artifact_path)


def get_artifact_dir(run: Run_ | Series | str) -> Path:
    uri = get_artifact_uri(run)
    return uri_to_path(uri)


def get_artifact_path(run: Run_ | Series | str, artifact_path: str | None = None) -> Path:
    artifact_dir = get_artifact_dir(run)
    return artifact_dir / artifact_path if artifact_path else artifact_dir


def load_config(run: Run_ | Series | str) -> DictConfig:
    run_id = get_run_id(run)
    return _load_config(run_id)


@cache
def _load_config(run_id: str) -> DictConfig:
    try:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=".hydra/config.yaml",
        )
    except OSError:
        return DictConfig({})

    return OmegaConf.load(path)  # type: ignore


def get_hydra_output_dir(run: Run_ | Series | str) -> Path:
    path = get_artifact_dir(run) / ".hydra/hydra.yaml"

    if path.exists():
        hc = OmegaConf.load(path)
        return Path(hc.hydra.runtime.output_dir)

    raise FileNotFoundError


def log_hydra_output_dir(run: Run_ | Series | str) -> None:
    output_dir = get_hydra_output_dir(run)
    run_id = run if isinstance(run, str) else run.info.run_id
    mlflow.log_artifacts(output_dir.as_posix(), run_id=run_id)
