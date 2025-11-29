from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydraflow

if TYPE_CHECKING:
    from mlflow.entities import Run


@dataclass
class Config:
    width: int = 0
    height: int = 0
    area: int = 0

    @staticmethod
    def update(cfg: Config) -> Config:
        if cfg.width > 0 and cfg.height > 0:
            cfg.area = cfg.width * cfg.height
        elif cfg.width > 0 and cfg.area > 0:
            cfg.height = cfg.area // cfg.width
        elif cfg.height > 0 and cfg.area > 0:
            cfg.width = cfg.area // cfg.height
        return cfg


@hydraflow.main(
    Config,
    chdir=True,
    update=Config.update,
    tracking_uri="sqlite:///mlflow.db",
)
def app(_run: Run, _cfg: Config):
    pass


if __name__ == "__main__":
    app()
