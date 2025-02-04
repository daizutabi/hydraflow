from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

import hydraflow


@dataclass
class Config:
    count: int = 0


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def app(cfg: Config):
    hydraflow.set_experiment()

    with hydraflow.start_run(cfg, chdir=True):
        Path("a.txt").write_text(str(cfg.count))


if __name__ == "__main__":
    app()
