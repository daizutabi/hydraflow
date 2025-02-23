from hydraflow.jobs.step import Step


def test_iter_batches():
    step = Step(args="a=1:3", batch="b=3,4 c=5,6")
    it = Step.iter_batches(step)
    assert next(it) == "b=3 c=5 a=1,2,3"
    assert next(it) == "b=3 c=6 a=1,2,3"
    assert next(it) == "b=4 c=5 a=1,2,3"
    assert next(it) == "b=4 c=6 a=1,2,3"


def test_iter_batches_pipe():
    step = Step(args="a=1:3", batch="b=3,4|c=5:7")
    it = Step.iter_batches(step)
    assert next(it) == "b=3,4 a=1,2,3"
    assert next(it) == "c=5,6,7 a=1,2,3"


def test_iter_batches_with_options():
    step = Step(args="a=1:3", batch="b=3,4", options="--opt1 --opt2")
    it = Step.iter_batches(step)
    assert next(it) == "--opt1 --opt2 b=3 a=1,2,3"
    assert next(it) == "--opt1 --opt2 b=4 a=1,2,3"
