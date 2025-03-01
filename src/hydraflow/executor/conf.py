from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Step:
    batch: str = ""
    args: str = ""
    configs: str = ""


@dataclass
class Job:
    name: str = ""
    run: str = ""
    call: str = ""
    configs: str = ""
    steps: list[Step] = field(default_factory=list)


@dataclass
class HydraflowConf:
    jobs: dict[str, Job] = field(default_factory=dict)
