from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlflow

import hydraflow

if TYPE_CHECKING:
    from mlflow.entities import Run


@dataclass
class Config:
    count: int = 1
    name: str = "a"


@hydraflow.main(Config, match_overrides=True)
def app(_run: Run, _cfg: Config):
    pass


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    app()
