from __future__ import annotations

from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore

import hydraflow


@dataclass
class Config:
    host: str = "localhost"
    port: int = 3306


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def app(cfg: Config):
    hydraflow.set_experiment()

    if hydraflow.list_runs().filter(cfg, override=True):
        return

    with hydraflow.start_run(cfg):
        pass


if __name__ == "__main__":
    app()
