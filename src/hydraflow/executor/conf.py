"""Configuration dataclasses for Hydraflow executor."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Set:
    """A set of commands or actions to be executed."""

    each: str = ""
    all: str = ""
    add: str = ""


@dataclass
class Job:
    """A job configuration containing various commands and sets."""

    name: str = ""
    run: str = ""
    call: str = ""
    submit: str = ""
    add: str = ""
    sets: list[Set] = field(default_factory=list)


@dataclass
class HydraflowConf:
    """Configuration for Hydraflow executor."""

    jobs: dict[str, Job] = field(default_factory=dict)
