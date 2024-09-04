from __future__ import annotations

from typing import TYPE_CHECKING

import joblib
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

if TYPE_CHECKING:
    from collections.abc import Iterable


def progress(
    *iterables: Iterable[int | tuple[int, int]],
    n_jobs: int = -1,
    task_name: str = "#{:0>3}",
    main_task_name: str = "main",
) -> None:
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        n = len(iterables)

        task_main = progress.add_task(main_task_name, total=None) if n > 1 else None
        tasks = [progress.add_task(task_name.format(i), start=False, total=None) for i in range(n)]

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

        if n > 1:
            it = (joblib.delayed(func)(i) for i in range(n))
            joblib.Parallel(n_jobs, prefer="threads")(it)

        else:
            func(0)
