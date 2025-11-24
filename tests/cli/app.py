from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlflow

import hydraflow

if TYPE_CHECKING:
    from mlflow.entities import Run

log = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for the example app.

    Attributes:
        count: Number of iterations to run
        name: Identifier for the run
    """

    count: int = 1
    name: str = "a"


@hydraflow.main(Config)
def app(_run: Run, _cfg: Config):
    """Example app demonstrating Hydraflow's basic functionality.

    This app shows how to:

    1. Define a configuration using dataclasses
    2. Use the Hydraflow decorator to integrate with MLflow
    3. Access the MLflow run instance and configuration

    Args:
        run: MLflow run instance for tracking metrics and parameters
        cfg: Configuration instance containing run parameters
    """
    # Start the run
    log.info("start")

    # Simulate some work
    time.sleep(0.2)

    # End the run
    log.info("end")


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    app()
