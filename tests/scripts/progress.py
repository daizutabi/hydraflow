from __future__ import annotations

import random
import time

from rich.progress import (
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
)

from hydraflow import multi_tasks_progress, parallel_progress


def task(total):
    for i in range(total or 90):
        if total is None:
            yield i
        else:
            yield i, total
        time.sleep(random.random() / 30)


def parallel_progress_test():
    def func(x: int) -> str:
        time.sleep(1)
        return f"result: {x}"

    it = range(12)

    columns = [
        SpinnerColumn(),
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ]

    results = parallel_progress(func, it, *columns, n_jobs=4)
    print(results)


def multi_tasks_progress_test(unknown_total: bool):
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

    multi_tasks_progress(tasks, *columns, n_jobs=4, **kwargs)


if __name__ == "__main__":
    parallel_progress_test()

    multi_tasks_progress_test(False)
    multi_tasks_progress_test(True)
    multi_tasks_progress([task(100)])
    multi_tasks_progress([task(None)], description="unknown")

    desc = "transient"
    multi_tasks_progress([task(100), task(None)], main_description=desc, transient=True)
    multi_tasks_progress([task(100)], description=desc, transient=True)
