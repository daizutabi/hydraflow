import logging
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore

import hydraflow

log = logging.getLogger(__name__)


@dataclass
class Config:
    width: int = 1024
    height: int = 768


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def app(cfg: Config) -> None:
    hydraflow.set_experiment()

    with hydraflow.start_run(cfg):
        log.info(f"{cfg.width=}, {cfg.height=}")


if __name__ == "__main__":
    app()
