import pytest
from mlflow.entities import Run

from hydraflow.run_collection import RunCollection

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(scope="module")
def rc(collect):
    args = ["host=a"]
    return collect("param/params.py", args)


@pytest.fixture(scope="module")
def run(rc: RunCollection):
    return rc.first()


def test_get_params_str(run: Run):
    from hydraflow.param import get_params

    assert get_params(run, "host") == ("a",)


def test_get_params_list(run: Run):
    from hydraflow.param import get_params

    assert get_params(run, ["host"], ["port"]) == ("a", "3306")


def test_get_values(run: Run):
    from hydraflow.param import get_values

    assert get_values(run, ["host", "port"], [str, int]) == ("a", 3306)
