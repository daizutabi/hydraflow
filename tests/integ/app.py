from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

import hydra
from hydra.core.config_store import ConfigStore

import hydraflow

log = logging.getLogger(__name__)


class E(Enum):
    A = auto()
    B = auto()


@dataclass
class B:
    z: float = 0.0


@dataclass
class A:
    y: str = "y"
    b: B = field(default_factory=B)
    e: E = E.A


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
    log.info(cfg)
    log.info(hydraflow.get_overrides())
    log.info(hydraflow.select_overrides(cfg))
    log.info(rc.filter(cfg, override=True))
    for r in rc:
        log.info(r.data.params)
        log.info(hydraflow.load_config(r))

    cfg.y = 2 * cfg.x
    with hydraflow.start_run(cfg):
        pass


if __name__ == "__main__":
    app()
