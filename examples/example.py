from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydraflow

if TYPE_CHECKING:
    from mlflow.entities import Run

# pyright: reportUnknownArgumentType=false

logger = logging.getLogger(__name__)


@dataclass
class Config:
    width: int = 1024
    height: int = 768


@hydraflow.main(Config)
def app(run: Run, cfg: Config) -> None:
    logger.info(run.info.run_id)
    logger.info(cfg)


if __name__ == "__main__":
    app()
