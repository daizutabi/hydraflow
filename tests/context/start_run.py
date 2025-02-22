from __future__ import annotations

from dataclasses import dataclass

import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

import hydraflow


@dataclass
class Config:
    name: str = "a"


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(config_name="config", version_base=None)
def app(cfg: Config):
    hc = HydraConfig.get()
    mlflow.set_experiment(hc.job.name)

    with hydraflow.start_run(cfg) as run:
        mlflow.log_text(cfg.name, "1.txt")

    with hydraflow.start_run(cfg, run_id=run.info.run_id):  # Skip log config
        mlflow.log_text(cfg.name * 2, "2.txt")


if __name__ == "__main__":
    app()
