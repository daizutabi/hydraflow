from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

import hydraflow


@dataclass
class Config:
    count: int = 0


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(config_name="config", version_base=None)
def app(cfg: Config):
    hc = HydraConfig.get()
    mlflow.set_experiment(hc.job.name)

    with hydraflow.start_run(chdir=True):
        Path("a.txt").write_text(str(cfg.count))


if __name__ == "__main__":
    app()
