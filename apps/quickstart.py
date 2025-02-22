import logging
from dataclasses import dataclass

import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

import hydraflow

log = logging.getLogger(__name__)


@dataclass
class Config:
    width: int = 1024
    height: int = 768


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_name="config", version_base=None)
def app(cfg: Config) -> None:
    hc = HydraConfig.get()
    mlflow.set_experiment(hc.job.name)

    with hydraflow.start_run(cfg):
        log.info(f"{cfg.width=}, {cfg.height=}")


if __name__ == "__main__":
    app()
