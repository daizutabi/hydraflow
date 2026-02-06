from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

import hydraflow

logger = logging.getLogger(__name__)


@dataclass
class Config:
    count: int = 0


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(config_name="config", version_base=None)
def app(cfg: Config) -> None:
    hc = HydraConfig.get()
    mlflow.set_experiment(hc.job.name)

    with hydraflow.start_run():
        logger.info("logger.info")

        mlflow.log_text("mlflow.log_text", "text.log")

        output_dir = Path(hc.runtime.output_dir)
        (output_dir / "text.log").write_text("write_text")
        (output_dir / "dir.log").mkdir()

        assert cfg.count == 200


if __name__ == "__main__":
    app()
