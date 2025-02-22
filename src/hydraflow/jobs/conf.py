from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Step:
    run: str = ""
    args: str = ""
    batch: str = ""


@dataclass
class Job:
    run: str = ""
    steps: list[Step] = field(default_factory=list)


@dataclass
class HydraflowConf:
    run: str = ""
    jobs: dict[str, Job] = field(default_factory=dict)
