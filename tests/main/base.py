from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import hydraflow

if TYPE_CHECKING:
    from mlflow.entities import Run


@dataclass
class Config:
    count: int = 0


@hydraflow.main(Config)
def app(run: Run, cfg: Config):
    path = hydraflow.get_artifact_dir() / "a.txt"
    path.write_text(f"{run.info.run_id},{cfg.count}")
    path = hydraflow.get_artifact_dir() / "b.txt"
    path.write_text(f"{Path.cwd().absolute().as_posix()}")


if __name__ == "__main__":
    app()
