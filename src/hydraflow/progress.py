from __future__ import annotations

from typing import TYPE_CHECKING

import joblib
from rich.progress import Progress

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rich.progress import ProgressColumn


def multi_task_progress(
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
            progress.add_task(description.format(i), start=False, total=None) for i in range(n)
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


if __name__ == "__main__":
    import random
    import time

    from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn

    from hydraflow.progress import multi_task_progress

    def task(total):
        for i in range(total or 90):
            if total is None:
                yield i
            else:
                yield i, total
            time.sleep(random.random() / 30)

    def multi_task_progress_test(unknown_total: bool):
        tasks = [task(random.randint(80, 100)) for _ in range(4)]
        if unknown_total:
            tasks = [task(None), *tasks, task(None)]

        columns = [
            SpinnerColumn(),
            *Progress.get_default_columns(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ]

        kwargs = {}
        if unknown_total:
            kwargs["main_description"] = "unknown"

        multi_task_progress(tasks, *columns, n_jobs=4, **kwargs)

    multi_task_progress_test(False)
    multi_task_progress_test(True)
    multi_task_progress([task(100)])
    multi_task_progress([task(None)], description="unknown")
    multi_task_progress([task(100), task(None)], main_description="transient", transient=True)
    multi_task_progress([task(100)], description="transient", transient=True)
