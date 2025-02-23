"""A step in a job."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from .parser import parse


@dataclass
class Step:
    """A step in a job."""

    run: str = ""
    args: str = ""
    batch: str = ""

    @classmethod
    def parse(cls, step: Self) -> list[str]:
        return []
