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
    age: int = 1
    height: float = 1.7


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(config_name="config", version_base=None)
def app(cfg: Config):
    hc = HydraConfig.get()
    mlflow.set_experiment(hc.job.name)

    with hydraflow.start_run(cfg):
        hydra_output_dir = hydraflow.get_hydra_output_dir()
        mlflow.log_text(hydra_output_dir.as_posix(), "hydra_output_dir.txt")


if __name__ == "__main__":
    app()
