"""A step in a job."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from .parser import collect, expand

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class Step:
    """A step in a job."""

    args: str = ""
    batch: str = ""
    options: str = ""

    @classmethod
    def iter_batches(cls, step: Self) -> Iterator[str]:
        """Iterate over combinations generated from parsed arguments.

        Generate all possible combinations of arguments by parsing and
        expanding each one, yielding them as an iterator.

        Args:
            step (Step): The step to parse.

        Yields:
            list[str]: a list of the parsed argument combinations.

        """
        args = collect(step.args)
        options = [o for o in step.options.split(" ") if o]

        for batch in expand(step.batch):
            yield shlex.join([*options, *batch, *args])
