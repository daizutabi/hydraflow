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


def test_parallel_progress(**kwargs):
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

    parallel_progress(func, it, *columns, n_jobs=-1, **kwargs)


def task(total):
    for i in range(total or 90):
        if total is None:
            yield i
        else:
            yield i, total
        time.sleep(random.random() / 30)


def test_multi_tasks_progress(total: bool, **kwargs):
    tasks = (task(random.randint(80, 100)) for _ in range(4))
    if total:
        tasks = (task(None), *list(tasks)[:2], task(None))

    columns = [
        SpinnerColumn(),
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ]

    if total:
        kwargs["main_description"] = "unknown"

    multi_tasks_progress(tasks, *columns, n_jobs=4, **kwargs)


if __name__ == "__main__":
    test_parallel_progress(description="parallel")
    test_parallel_progress(transient=True)
    test_multi_tasks_progress(False)
    test_multi_tasks_progress(True, transient=False)
    test_multi_tasks_progress(False, transient=True)
