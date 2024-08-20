from __future__ import annotations

import logging
from dataclasses import dataclass

import hydra
import mlflow
from hydra.core.config_store import ConfigStore

from hydraflow.context import log_run

log = logging.getLogger(__name__)


@dataclass
class MySQLConfig:
    host: str = "localhost"
    port: int = 3306


cs = ConfigStore.instance()
cs.store(name="config", node=MySQLConfig)


@hydra.main(version_base=None, config_name="config")
def app(cfg: MySQLConfig):
    mlflow.set_experiment("log_run")
    with mlflow.start_run(), log_run(cfg) as info:
        log.info(f"START, {cfg.host}, {cfg.port} ")
        mlflow.log_text(info.artifact_dir.as_posix(), "artifact_dir.txt")
        mlflow.log_text(info.output_dir.as_posix(), "output_dir.txt")
        (info.artifact_dir / "a.txt").write_text("abc")
        log.info("END")


if __name__ == "__main__":
    app()
