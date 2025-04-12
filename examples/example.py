import logging
from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from mlflow.entities import Run

import hydraflow

log = logging.getLogger(__name__)


@dataclass
class Config:
    width: int = 1024
    height: int = 768


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydraflow.main(Config)
def app(run: Run, cfg: Config) -> None:
    log.info(run.info.run_id)
    log.info(cfg)


if __name__ == "__main__":
    app()
