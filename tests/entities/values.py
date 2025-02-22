from __future__ import annotations

from dataclasses import dataclass, field

import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

import hydraflow


@dataclass
class Config:
    host: str = "localhost"
    port: int = 3306
    x: float = 1e-8
    y: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(config_name="config", version_base=None)
def app(cfg: Config):
    hc = HydraConfig.get()
    mlflow.set_experiment(hc.job.name)

    with hydraflow.start_run(cfg):
        pass


if __name__ == "__main__":
    app()
