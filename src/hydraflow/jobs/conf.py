from __future__ import annotations

from dataclasses import dataclass, field

from .step import Step


@dataclass
class Job:
    run: str = ""
    steps: list[Step] = field(default_factory=list)


@dataclass
class HydraflowConf:
    jobs: dict[str, Job] = field(default_factory=dict)
