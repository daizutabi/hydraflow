from __future__ import annotations

from dataclasses import dataclass, field

import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

import hydraflow


@dataclass
class Data:
    x: list[int] = field(default_factory=lambda: [1, 2, 3])
    y: list[int] = field(default_factory=lambda: [4, 5, 6])


@dataclass
class Config:
    host: str = "localhost"
    port: int = 3306
    data: Data = field(default_factory=Data)


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(config_name="config", version_base=None)
def app(cfg: Config):
    hc = HydraConfig.get()
    mlflow.set_experiment(hc.job.name)

    with hydraflow.start_run(cfg):
        pass


if __name__ == "__main__":
    app()
