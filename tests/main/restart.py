from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import hydraflow

if TYPE_CHECKING:
    from mlflow.entities import Run


@dataclass
class A:
    a: int = 1


@dataclass
class Config:
    count: int = 0
    x: list[float] = field(default_factory=lambda: [1, 2, 3])
    a: A = field(default_factory=A)


@hydraflow.main(Config, chdir=True, rerun_finished=False)
def app(run: Run, cfg: Config):
    file = Path("a.txt")
    text = file.read_text() if file.exists() else ""
    file.write_text(text + f"{cfg.count}")


if __name__ == "__main__":
    app()
