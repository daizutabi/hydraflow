from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import hydra
import mlflow
from hydra.core.config_store import ConfigStore

import hydraflow

log = logging.getLogger(__name__)


@dataclass
class MySQLConfig:
    host: str = "localhost"
    port: int = 3306


cs = ConfigStore.instance()
cs.store(name="config", node=MySQLConfig)


@hydra.main(version_base=None, config_name="config")
def app(cfg: MySQLConfig):
    hydraflow.set_experiment(prefix="_", suffix="_")
    with hydraflow.start_run(cfg):
        log.info(f"START, {cfg.host}, {cfg.port} ")

        artifact_dir = hydraflow.get_artifact_dir()
        output_dir = hydraflow.get_hydra_output_dir()

        mlflow.log_text("A " + artifact_dir.as_posix(), "artifact_dir.txt")
        mlflow.log_text("B " + output_dir.as_posix(), "output_dir.txt")

        with hydraflow.watch(callback, ignore_patterns=["b.txt"]):
            (artifact_dir / "a.txt").write_text("abc")
            time.sleep(0.1)

        mlflow.log_metric("m", cfg.port + 1, 1)
        if cfg.host == "x":
            mlflow.log_metric("m", cfg.port + 10, 2)

        log.info("END")


def callback(path: Path):
    log.info(f"WATCH, {path.as_posix()}")
    m = len(path.read_text())  # len("abc") == 3
    mlflow.log_metric("watch", m, 1, synchronous=True)


if __name__ == "__main__":
    app()
