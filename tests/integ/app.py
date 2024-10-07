from __future__ import annotations

import logging
from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore

import hydraflow

log = logging.getLogger(__name__)


@dataclass
class B:
    z: float = 0.0


@dataclass
class A:
    y: str = "y"
    b: B = field(default_factory=B)


@dataclass
class Config:
    x: int = 0
    y: int = 0
    a: A = field(default_factory=A)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def app(cfg: Config):
    hydraflow.set_experiment()
    rc = hydraflow.list_runs()
    log.info(rc)
    log.info(hydraflow.select_overrides(cfg))
    log.info(rc.filter(cfg, override=True))
    log.info(rc.filter(cfg, select=["x"]))
    log.info(rc.try_find_last(cfg, override=True))
    log.info(rc.try_find_last(cfg, select=["x"]))
    log.info(rc.filter(cfg))

    cfg.y = 2 * cfg.x
    with hydraflow.start_run(cfg):
        pass


if __name__ == "__main__":
    app()
