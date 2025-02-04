from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

import hydraflow


@dataclass
class Config:
    name: str = "a"


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def app(cfg: Config):
    hydraflow.set_experiment()

    with hydraflow.start_run(cfg) as run:
        with hydraflow.chdir_hydra_output():
            Path("a.txt").write_text("chdir_hydra_output")

        with hydraflow.chdir_artifact(run):
            Path("b.txt").write_text("chdir_artifact")

        if cfg.name == "b":
            raise ValueError

        if cfg.name == "c":
            sys.exit(1)

    with hydraflow.start_run(cfg, run_id=run.info.run_id):  # Skip log config
        pass


if __name__ == "__main__":
    app()
