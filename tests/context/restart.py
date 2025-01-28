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


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_name="config")
def app(cfg: Config):
    hydraflow.set_experiment()

    if run := hydraflow.list_runs().try_find(cfg, override=True):
        run_id = run.info.run_id
    else:
        run_id = None

    with hydraflow.start_run(cfg, run_id=run_id) as run:
        process(hydraflow.get_artifact_dir(run))


def process(path: Path):
    file = path / "a.txt"
    text = file.read_text() if file.exists() else ""
    file.write_text(text + "a")


if __name__ == "__main__":
    app()
