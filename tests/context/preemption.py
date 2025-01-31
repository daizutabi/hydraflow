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


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def app(cfg: Config):
    hydraflow.set_experiment()

    rc = hydraflow.list_runs()

    if rc.filter(cfg, status="finished", override=True):
        return

    if run := rc.try_find(cfg, override=True):
        run_id = run.info.run_id
    else:
        run_id = None

    with hydraflow.start_run(cfg, run_id=run_id) as run:
        log(hydraflow.get_artifact_dir(run))


def log(path: Path):
    file = path / "a.txt"
    text = file.read_text() if file.exists() else ""
    file.write_text(text + "a")


if __name__ == "__main__":
    app()
