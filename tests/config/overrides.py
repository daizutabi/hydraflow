from __future__ import annotations

from dataclasses import dataclass

import hydra
import mlflow
from hydra.core.config_store import ConfigStore

import hydraflow


@dataclass
class Config:
    name: str = "a"
    age: int = 1
    height: float = 1.7


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def app(cfg: Config):
    hydraflow.set_experiment()

    with hydraflow.start_run(cfg):
        overrides = hydraflow.select_overrides(Config(name="x", height=2))
        mlflow.log_text(str(overrides), "overrides.txt")


if __name__ == "__main__":
    app()
