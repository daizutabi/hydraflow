from __future__ import annotations

from dataclasses import dataclass

import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

import hydraflow


@dataclass
class Config:
    host: str = "localhost"
    port: int = 3306


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(config_name="config", version_base=None)
def app(cfg: Config):
    hc = HydraConfig.get()
    e = mlflow.set_experiment(hc.job.name)

    if hydraflow.list_runs(e.name).filter(cfg):
        return

    with hydraflow.start_run(cfg):
        pass


if __name__ == "__main__":
    app()
