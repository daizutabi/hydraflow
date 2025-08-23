"""Provide a progress bar for parallel task execution.

This module defines the `Progress` class, which provides a visual
progress bar for tracking the execution of parallel tasks. It integrates
with the `joblib` library to display the progress of tasks being executed
in parallel, allowing users to monitor the completion status in real-time.

The `Progress` class can be customized to show different columns of
information, such as the elapsed time and the number of completed tasks.
It also provides methods to start and stop the progress display, as well
as to update the progress based on the number of completed tasks.

Example:
    ```python
    from hydraflow.utils.progress import Progress
    from joblib import Parallel, delayed

    with Progress(*Progress.get_default_columns()) as progress:
        Parallel(n_jobs=4)(delayed(function)(x) for x in iterable)
    ```

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from joblib.parallel import Parallel
from rich.progress import MofNCompleteColumn, SpinnerColumn, TimeElapsedColumn
from rich.progress import Progress as Super

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.progress import ProgressColumn


# https://github.com/jonghwanhyeon/joblib-progress/blob/main/joblib_progress/__init__.py
class Progress(Super):
    """A progress bar for tracking parallel task execution.

    This class extends the `rich.progress.Progress` class to provide
    a visual progress bar specifically designed for monitoring the
    execution of tasks running in parallel using the `joblib` library.
    It allows users to see the completion status of tasks in real-time
    and can be customized to display various columns of information.
    """

    _print_progress: Callable[[Parallel], None] | None = None

    def start(self) -> None:
        """Start the progress display."""
        super().start()

        self._print_progress = Parallel.print_progress

        def _update(parallel: Parallel) -> None:
            update(self, parallel)

        Parallel.print_progress = _update  # pyright: ignore[reportAttributeAccessIssue]

    def stop(self) -> None:
        """Stop the progress display."""
        if self._print_progress:
            Parallel.print_progress = self._print_progress  # pyright: ignore[reportAttributeAccessIssue]

        super().stop()

    @classmethod
    def get_default_columns(cls) -> tuple[ProgressColumn, ...]:
        """Get the default columns used for a new Progress instance."""
        return (
            SpinnerColumn(),
            TimeElapsedColumn(),
            *Super.get_default_columns(),
            MofNCompleteColumn(),
        )


def update(progress: Progress, parallel: Parallel) -> None:
    """Update the progress bar."""
    if progress.task_ids:
        task_id = progress.task_ids[-1]
    else:
        task_id = progress.add_task("", total=None)

    progress.update(task_id, completed=parallel.n_completed_tasks, refresh=True)

    if progress._print_progress:  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        progress._print_progress(parallel)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
