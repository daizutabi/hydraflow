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


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def app(cfg: Config):
    hydraflow.set_experiment()

    with hydraflow.start_run(cfg):
        hydra_output_dir = hydraflow.get_hydra_output_dir()
        mlflow.log_text(hydra_output_dir.as_posix(), "hydra_output_dir.txt")

        overrides = hydraflow.get_overrides()
        mlflow.log_text(str(overrides), "overrides.txt")


if __name__ == "__main__":
    app()
