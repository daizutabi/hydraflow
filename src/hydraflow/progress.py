# def parallel_save(uris: Iterable[list[str]], n_jobs: int = -1) -> None:
#     import joblib
#     from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

#     from magsim import load_runs

#     uris = list(uris)

#     with Progress(
#         SpinnerColumn(),
#         *Progress.get_default_columns(),
#         TimeElapsedColumn(),
#     ) as progress:
#         n = len(uris)
#         task_main = progress.add_task("all (----/----)", total=None) if n > 1 else None
#         tasks = [
#             progress.add_task(f"#{i:0>2} (----/----)", start=False, total=None)
#             for i in range(n)
#         ]
#         total = {}
#         completed = {}

#         def save(i: int) -> None:
#             runs = load_runs(uris[i])
#             video = Video.create(runs, "m", width=256)
#             total[i] = len(video)
#             completed[i] = 0
#             progress.update(tasks[i], total=total[i])
#             progress.start_task(tasks[i])
#             saving = video.saving(f"{video.quantity}.mp4", output_dir=video.directory)

#             for _ in saving:
#                 completed[i] += 1
#                 description = f"#{i:0>2} ({completed[i]:0>4}/{total[i]:0>4})"
#                 progress.update(
#                     tasks[i],
#                     description=description,
#                     completed=completed[i],
#                 )
#                 if task_main is not None:
#                     c = sum(completed.values())
#                     t = sum(total.values())
#                     description = f"all ({c:0>4}/{t:0>4})"
#                     progress.update(
#                         task_main,
#                         description=description,
#                         total=t,
#                         completed=c,
#                     )

#         if n > 1:
#             it = (joblib.delayed(save)(i) for i in range(n))
#             joblib.Parallel(n_jobs, prefer="threads")(it)

#         else:
#             save(0)
