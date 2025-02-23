from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Step:
    args: str = ""
    batch: str = ""
    options: str = ""


@dataclass
class Job:
    name: str = ""
    run: str = ""
    call: str = ""
    steps: list[Step] = field(default_factory=list)


@dataclass
class HydraflowConf:
    jobs: dict[str, Job] = field(default_factory=dict)
