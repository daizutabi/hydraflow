from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import mlflow
import numpy as np
from mlflow.entities.run import Run
from mlflow.tracking import artifact_utils
from omegaconf import DictConfig, OmegaConf

from hydraflow.config import iter_params
from hydraflow.util import uri_to_path

if TYPE_CHECKING:
    from typing import Any

    from pandas import DataFrame, Series


@overload
def filter_by_config(runs: list[Run], config: object) -> list[Run]: ...


@overload
def filter_by_config(runs: DataFrame, config: object) -> DataFrame: ...


def filter_by_config(runs: list[Run] | DataFrame, config: object):
    if isinstance(runs, list):
        return filter_by_config_list(runs, config)

    return filter_by_config_dataframe(runs, config)


def _is_equal(run: Run, key: str, value: Any) -> bool:
    param = run.data.params.get(key, value)
    if param is None:
        return False

    return type(value)(param) == value


def filter_by_config_list(runs: list[Run], config: object) -> list[Run]:
    for key, value in iter_params(config):
        runs = [run for run in runs if _is_equal(run, key, value)]

    return runs


def filter_by_config_dataframe(runs: DataFrame, config: object) -> DataFrame:
    index = np.ones(len(runs), dtype=bool)

    for key, value in iter_params(config):
        name = f"params.{key}"
        if name in runs:
            series = runs[name]
            is_value = -series.isna()
            param = series.fillna(value).astype(type(value))
            index &= is_value & (param == value)

    return runs[index]


@overload
def get_by_config(runs: list[Run], config: object) -> Run: ...


@overload
def get_by_config(runs: DataFrame, config: object) -> Series: ...


def get_by_config(runs: list[Run] | DataFrame, config: object):
    runs = filter_by_config(runs, config)

    if len(runs) == 1:
        return runs[0] if isinstance(runs, list) else runs.iloc[0]

    msg = f"filtered runs has not length of 1.: {len(runs)}"
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


def get_run_id(run: Run | Series | str) -> str:
    if isinstance(run, Run):
        return run.info.run_id
    if isinstance(run, str):
        return run
    return run.run_id


def get_artifact_uri(run: Run | Series | str, artifact_path: str | None = None) -> str:
    if isinstance(run, Run):
        uri = run.info.artifact_uri
    elif isinstance(run, str):
        uri = artifact_utils.get_artifact_uri(run_id=run)
    else:
        uri = run.artifact_uri

    if artifact_path:
        uri = f"{uri}/{artifact_path}"

    return uri  # type: ignore


def get_artifact_dir(run: Run | Series | str) -> Path:
    uri = get_artifact_uri(run)
    return uri_to_path(uri)


def get_artifact_path(
    run: Run | Series | str,
    artifact_path: str | None = None,
) -> Path:
    artifact_dir = get_artifact_dir(run)
    return artifact_dir / artifact_path if artifact_path else artifact_dir


def load_config(run: Run | Series | str, output_subdir: str = ".hydra") -> DictConfig:
    run_id = get_run_id(run)

    try:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=f"{output_subdir}/config.yaml",
        )
    except OSError:
        return DictConfig({})

    return OmegaConf.load(path)  # type: ignore


def get_hydra_output_dir(run: Run | Series | str) -> Path:
    path = get_artifact_dir(run) / ".hydra/hydra.yaml"

    if path.exists():
        hc = OmegaConf.load(path)
        return Path(hc.hydra.runtime.output_dir)

    raise FileNotFoundError


def log_hydra_output_dir(run: Run | Series | str) -> None:
    output_dir = get_hydra_output_dir(run)
    run_id = run if isinstance(run, str) else run.info.run_id
    mlflow.log_artifacts(output_dir.as_posix(), run_id=run_id)
