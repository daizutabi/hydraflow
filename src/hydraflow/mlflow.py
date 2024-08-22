from __future__ import annotations

import mlflow
from hydra.core.hydra_config import HydraConfig

from hydraflow.config import iter_params


def set_experiment(prefix: str = "", suffix: str = "", uri: str | None = None) -> None:
    if uri:
        mlflow.set_tracking_uri(uri)

    hc = HydraConfig.get()
    name = f"{prefix}{hc.job.name}{suffix}"
    mlflow.set_experiment(name)


def log_params(config: object, *, synchronous: bool | None = None) -> None:
    for key, value in iter_params(config):
        mlflow.log_param(key, value, synchronous=synchronous)
