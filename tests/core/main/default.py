from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import hydraflow

if TYPE_CHECKING:
    from mlflow.entities import Run


@dataclass
class Config:
    count: int = 1
    name: str = "a"


@hydraflow.main(Config)
def app(run: Run, cfg: Config) -> None:
    path = hydraflow.get_artifact_dir(run) / "a.txt"
    path.write_text(f"{cfg.count}")
    path = hydraflow.get_artifact_dir(run) / "b.txt"
    path.write_text(f"{Path.cwd().absolute().as_posix()}")


if __name__ == "__main__":
    app()
