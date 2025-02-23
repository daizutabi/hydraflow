from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydraflow

if TYPE_CHECKING:
    from mlflow.entities import Run


@dataclass
class Config:
    count: int = 1
    name: str = "a"


@hydraflow.main(Config)
def app(run: Run, cfg: Config):
    pass


if __name__ == "__main__":
    app()
