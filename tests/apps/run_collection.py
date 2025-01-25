from __future__ import annotations

from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore

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


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def app(cfg: Config):
    hydraflow.set_experiment()

    with hydraflow.start_run(cfg):
        pass


if __name__ == "__main__":
    app()
