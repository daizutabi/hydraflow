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


@hydraflow.main(Config, chdir=True, force_new_run=True)
def app(_run: Run, cfg: Config):
    file = Path("a.txt")
    text = file.read_text() if file.exists() else ""
    file.write_text(text + f"{cfg.count}")


if __name__ == "__main__":
    app()
