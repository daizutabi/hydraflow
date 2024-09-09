"""
Module for managing progress tracking in parallel processing using Joblib
and Rich's Progress bar.

Provide context managers and functions to facilitate the execution
of tasks in parallel while displaying progress updates.

The following key components are provided:

- JoblibProgress: A context manager for tracking progress with Rich's Progress
    bar.
- parallel_progress: A function to execute a given function in parallel over
    an iterable with progress tracking.
- multi_tasks_progress: A function to render auto-updating progress bars for
    multiple tasks concurrently.

Usage:
    Import the necessary functions and use them to manage progress in your
    parallel processing tasks.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, TypeVar

import joblib
from rich.progress import Progress

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from rich.progress import ProgressColumn


# https://github.com/jonghwanhyeon/joblib-progress/blob/main/joblib_progress/__init__.py
@contextmanager
def JoblibProgress(
    *columns: ProgressColumn | str,
    description: str | None = None,
    total: int | None = None,
    **kwargs,
) -> Iterator[Progress]:
    """
    Context manager for tracking progress using Joblib with Rich's Progress bar.

    Args:
        *columns (ProgressColumn | str): Columns to display in the progress bar.
        description (str | None, optional): A description for the progress task.
            Defaults to None.
        total (int | None, optional): The total number of tasks. If None, it will
            be determined automatically.
        **kwargs: Additional keyword arguments passed to the Progress instance.

    Yields:
        Progress: A Progress instance for managing the progress bar.

    Example:
        with JoblibProgress("task", total=100) as progress:
            # Your parallel processing code here
    """
    if not columns:
        columns = Progress.get_default_columns()

    progress = Progress(*columns, **kwargs)

    if description is None:
        description = "Processing..."

    task_id = progress.add_task(description, total=total)
    print_progress = joblib.parallel.Parallel.print_progress

    def update_progress(self: joblib.parallel.Parallel):
        progress.update(task_id, completed=self.n_completed_tasks, refresh=True)
        return print_progress(self)

    try:
        joblib.parallel.Parallel.print_progress = update_progress
        progress.start()
        yield progress

    finally:
        progress.stop()
        joblib.parallel.Parallel.print_progress = print_progress


T = TypeVar("T")
U = TypeVar("U")


def parallel_progress(
    func: Callable[[T], U],
    iterable: Iterable[T],
    *columns: ProgressColumn | str,
    n_jobs: int = -1,
    description: str | None = None,
    **kwargs,
) -> list[U]:
    """
    Execute a function in parallel over an iterable with progress tracking.

    Args:
        func (Callable[[T], U]): The function to execute on each item in the
            iterable.
        iterable (Iterable[T]): An iterable of items to process.
        *columns (ProgressColumn | str): Additional columns to display in the
            progress bar.
        n_jobs (int, optional): The number of jobs to run in parallel.
            Defaults to -1 (all processors).
        description (str | None, optional): A description for the progress bar.
            Defaults to None.
        **kwargs: Additional keyword arguments passed to the Progress instance.

    Returns:
        list[U]: A list of results from applying the function to each item in
        the iterable.
    """
    iterable = list(iterable)
    total = len(iterable)

    with JoblibProgress(*columns, description=description, total=total, **kwargs):
        it = (joblib.delayed(func)(x) for x in iterable)
        return joblib.Parallel(n_jobs=n_jobs)(it)  # type: ignore


def multi_tasks_progress(
    iterables: Iterable[Iterable[int | tuple[int, int]]],
    *columns: ProgressColumn | str,
    n_jobs: int = -1,
    description: str = "#{:0>3}",
    main_description: str = "main",
    transient: bool | None = None,
    **kwargs,
) -> None:
    """
    Render auto-updating progress bars for multiple tasks concurrently.

    Args:
        iterables (Iterable[Iterable[int | tuple[int, int]]]): A collection of
            iterables, each representing a task. Each iterable can yield
            integers (completed) or tuples of integers (completed, total).
        *columns (ProgressColumn | str): Additional columns to display in the
            progress bars.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to
            -1, which means using all processors.
        description (str, optional): Format string for describing tasks. Defaults to
            "#{:0>3}".
        main_description (str, optional): Description for the main task.
            Defaults to "main".
        transient (bool | None, optional): Whether to remove the progress bar
            after completion. Defaults to None.
        **kwargs: Additional keyword arguments passed to the Progress instance.

    Returns:
        None
    """
    if not columns:
        columns = Progress.get_default_columns()

    iterables = list(iterables)

    with Progress(*columns, transient=transient or False, **kwargs) as progress:
        n = len(iterables)

        task_main = progress.add_task(main_description, total=None) if n > 1 else None
        tasks = [
            progress.add_task(description.format(i), start=False, total=None)
            for i in range(n)
        ]

        total = {}
        completed = {}

        def func(i: int) -> None:
            completed[i] = 0
            total[i] = None
            progress.start_task(tasks[i])

            for index in iterables[i]:
                if isinstance(index, tuple):
                    completed[i], total[i] = index[0] + 1, index[1]
                else:
                    completed[i] = index + 1

                progress.update(tasks[i], total=total[i], completed=completed[i])
                if task_main is not None:
                    if all(t is not None for t in total.values()):
                        t = sum(total.values())
                    else:
                        t = None
                    c = sum(completed.values())
                    progress.update(task_main, total=t, completed=c)

            if transient or n > 1:
                progress.remove_task(tasks[i])

        if n > 1:
            it = (joblib.delayed(func)(i) for i in range(n))
            joblib.Parallel(n_jobs, prefer="threads")(it)

        else:
            func(0)
