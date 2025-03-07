from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hydra.core.config_store import ConfigStore

import hydraflow

if TYPE_CHECKING:
    from mlflow.entities import Run

log = logging.getLogger(__name__)


@dataclass
class Config:
    width: int = 1024
    height: int = 768


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydraflow.main(Config)
def app(run: Run, cfg: Config) -> None:
    print(run.info.run_id)
    print(cfg)


if __name__ == "__main__":
    app()
