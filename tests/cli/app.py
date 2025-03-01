from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydraflow

if TYPE_CHECKING:
    from mlflow.entities import Run

log = logging.getLogger(__name__)


@dataclass
class Config:
    count: int = 1
    name: str = "a"


@hydraflow.main(Config)
def app(run: Run, cfg: Config):
    log.info("start")
    time.sleep(0.2)
    log.info("end")


if __name__ == "__main__":
    app()
