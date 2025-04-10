from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Set:
    each: str = ""
    all: str = ""
    add: str = ""


@dataclass
class Job:
    name: str = ""
    run: str = ""
    call: str = ""
    submit: str = ""
    add: str = ""
    sets: list[Set] = field(default_factory=list)


@dataclass
class HydraflowConf:
    jobs: dict[str, Job] = field(default_factory=dict)
