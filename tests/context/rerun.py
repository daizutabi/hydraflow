from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydra
from hydra.core.config_store import ConfigStore

import hydraflow

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class Config:
    count: int = 0


ConfigStore.instance().store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def app(cfg: Config):
    hydraflow.set_experiment()

    run = hydraflow.list_runs().try_find(cfg, override=True)

    with hydraflow.start_run(cfg, run=run) as run:
        log(hydraflow.get_artifact_dir(run))


def log(path: Path):
    file = path / "a.txt"
    text = file.read_text() if file.exists() else ""
    file.write_text(text + "a")


if __name__ == "__main__":
    app()
