from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

import hydraflow

log = logging.getLogger(__name__)


@dataclass
class Config:
    count: int = 0


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def app(cfg: Config):
    hydraflow.set_experiment()

    run = hydraflow.list_runs().try_get(cfg, override=True)

    with hydraflow.start_run(cfg, run=run):
        log.info("second" if run else "first")
        log.info(cfg.count)

        output_dir = Path(HydraConfig.get().runtime.output_dir)
        (output_dir / "text.log").write_text("text\n")
        (output_dir / "dir.log").mkdir()


if __name__ == "__main__":
    app()
