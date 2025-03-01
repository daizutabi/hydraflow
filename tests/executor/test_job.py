import sys
from pathlib import Path

import pytest

from hydraflow.executor.conf import Job, Step


def test_iter_args():
    from hydraflow.executor.job import iter_args

    it = iter_args("b=3,4 c=5,6", "a=1:3")
    assert next(it) == ["b=3", "c=5", "a=1,2,3"]
    assert next(it) == ["b=3", "c=6", "a=1,2,3"]
    assert next(it) == ["b=4", "c=5", "a=1,2,3"]
    assert next(it) == ["b=4", "c=6", "a=1,2,3"]


def test_iter_args_pipe():
    from hydraflow.executor.job import iter_args

    it = iter_args("b=3,4|c=5:7", "a=1:3")
    assert next(it) == ["b=3,4", "a=1,2,3"]
    assert next(it) == ["c=5,6,7", "a=1,2,3"]


@pytest.fixture
def job():
    s1 = Step(batch="b=5,6", args="a=1:2")
    s2 = Step(batch="c=7,8", args="a=3:4")
    return Job(name="test", steps=[s1, s2])


@pytest.fixture
def batches(job: Job):
    from hydraflow.executor.job import iter_batches

    return list(iter_batches(job))


def test_sweep_dir(batches):
    assert all(x[-1].startswith("hydra.sweep.dir=multirun/") for x in batches)
    assert all(len(x[-1].split("/")[-1]) == 26 for x in batches)


def test_job_name(batches):
    assert all(x[-2].startswith("hydra.job.name=test") for x in batches)


@pytest.mark.parametrize(("i", "x"), [(0, "b=5"), (1, "b=6"), (2, "c=7"), (3, "c=8")])
def test_batch_args(batches, i, x):
    assert batches[i][1] == x


@pytest.mark.parametrize(
    ("i", "x"),
    [(0, "a=1,2"), (1, "a=1,2"), (2, "a=3,4"), (3, "a=3,4")],
)
def test_sweep_args(batches, i, x):
    assert batches[i][-3] == x


def test_multirun_run(job: Job, tmp_path: Path):
    from hydraflow.executor.job import multirun

    path = tmp_path / "output.txt"
    file = Path(__file__).parent / "echo.py"

    job.run = f"python {file.as_posix()} {path.as_posix()}"
    multirun(job)
    assert path.read_text() == "b=5 a=1,2 b=6 a=1,2 c=7 a=3,4 c=8 a=3,4"


def test_multirun_run_error(job: Job):
    from hydraflow.executor.job import multirun

    job.run = "cmd /c exit 1" if sys.platform == "win32" else "false"
    with pytest.raises(RuntimeError):
        multirun(job)


def test_multirun_call(job: Job, capsys: pytest.CaptureFixture):
    from hydraflow.executor.job import multirun

    job.call = "typer.echo"
    multirun(job)
    out, _ = capsys.readouterr()
    assert "'b=5', 'a=1,2'" in out
    assert "'c=8', 'a=3,4'" in out


def test_multirun_call_args(job: Job, capsys: pytest.CaptureFixture):
    from hydraflow.executor.job import multirun

    job.call = "typer.echo a 'b c'"
    multirun(job)
    out, _ = capsys.readouterr()
    assert "['a', 'b c', '--multirun'," in out


def test_multirun_call_error(job: Job):
    from hydraflow.executor.job import multirun

    job.call = "hydraflow.executor.job.multirun"
    with pytest.raises(RuntimeError):
        multirun(job)


def test_multirun_call_invalid(job: Job):
    from hydraflow.executor.job import multirun

    job.call = "print"
    with pytest.raises(ValueError):
        multirun(job)


def test_multirun_call_not_found(job: Job):
    from hydraflow.executor.job import multirun

    job.call = "hydraflow.invalid"
    with pytest.raises(ValueError):
        multirun(job)


def test_to_text(job: Job):
    from hydraflow.executor.job import to_text

    job.call = "typer.echo"
    text = to_text(job)
    assert "call: typer.echo\n" in text
    assert "'b=5', 'a=1,2', 'hydra.job.name=test'" in text
    assert "'c=8', 'a=3,4', 'hydra.job.name=test'" in text
