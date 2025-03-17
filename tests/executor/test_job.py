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


def test_iter_runs(job: Job, tmp_path: Path):
    from hydraflow.executor.job import iter_batches, iter_runs

    path = tmp_path / "output.txt"
    file = Path(__file__).parent / "echo.py"

    args = [file.as_posix(), path.as_posix()]
    x = list(iter_runs("python", args, iter_batches(job)))
    assert path.read_text() == "b=5 a=1,2 b=6 a=1,2 c=7 a=3,4 c=8 a=3,4"
    assert x[0].completed == 1
    assert x[0].result.returncode == 0
    assert x[1].completed == 2
    assert x[1].result.returncode == 0
    assert x[2].completed == 3
    assert x[2].result.returncode == 0


def test_iter_calls(job: Job, capsys: pytest.CaptureFixture):
    from hydraflow.executor.job import iter_batches, iter_calls

    x = list(iter_calls("typer.echo", [], iter_batches(job)))
    out, _ = capsys.readouterr()
    assert "'b=5', 'a=1,2'" in out
    assert "'c=8', 'a=3,4'" in out
    assert x[0].completed == 1
    assert x[1].completed == 2
    assert x[2].completed == 3


def test_iter_calls_args(job: Job, capsys: pytest.CaptureFixture):
    from hydraflow.executor.job import iter_batches, iter_calls

    job.call = "typer.echo a 'b c'"
    list(iter_calls("typer.echo", ["a", "b c"], iter_batches(job)))
    out, _ = capsys.readouterr()
    assert "['a', 'b c', '--multirun'," in out


def test_submit(job: Job, capsys: pytest.CaptureFixture):
    from hydraflow.executor.job import iter_batches, submit

    submit("typer.echo", ["a"], iter_batches(job))
    out, _ = capsys.readouterr()
    assert out.startswith("[['a', '--multirun', 'b=5', 'a=1,2', 'hydra.job.name=test'")
    assert "], ['a', '--multirun', 'b=6', 'a=1,2', 'hydra" in out
    assert "], ['a', '--multirun', 'c=7', 'a=3,4', 'hydra" in out
    assert "], ['a', '--multirun', 'c=8', 'a=3,4', 'hydra" in out


def test_get_callable_error():
    from hydraflow.executor.job import get_callable

    with pytest.raises(ValueError):
        get_callable("print")


def test_get_callable_not_found():
    from hydraflow.executor.job import get_callable

    with pytest.raises(ValueError):
        get_callable("hydraflow.invalid")


def test_to_text(job: Job):
    from hydraflow.executor.job import to_text

    job.call = "typer.echo"
    text = to_text(job)
    assert "call: typer.echo\n" in text
    assert "'b=5', 'a=1,2', 'hydra.job.name=test'" in text
    assert "'c=8', 'a=3,4', 'hydra.job.name=test'" in text
