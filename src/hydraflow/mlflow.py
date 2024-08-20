from __future__ import annotations

import mlflow
from hydra.core.hydra_config import HydraConfig

from hydraflow.config import iter_params


def set_experiment() -> None:
    hc = HydraConfig.get()
    mlflow.set_tracking_uri("")
    mlflow.set_experiment(hc.job.name)


def log_params(config: object, *, synchronous: bool | None = None) -> None:
    for key, value in iter_params(config):
        mlflow.log_param(key, value, synchronous=synchronous)
