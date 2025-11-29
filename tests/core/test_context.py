from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import pytest
from hydra.conf import HydraConf

from hydraflow.core.context import log_run, start_run
from hydraflow.core.io import get_artifact_dir

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture(scope="module", autouse=True, params=["mlruns", "mlflow.db"])
def setup(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
):
    curdir = Path.cwd()
    tmpdir = tmp_path_factory.mktemp(request.param)
    os.chdir(tmpdir)
    assert isinstance(request.param, str)

    if "." in request.param:
        db = (tmpdir / request.param).as_posix()
        uri = f"sqlite:///{db}"
    else:
        uri = tmpdir / request.param

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("e1")

    yield

    os.chdir(curdir)


@pytest.fixture(autouse=True)
def hc(mocker: MockerFixture, tmp_path: Path) -> HydraConf:
    hc = HydraConf()
    hc.runtime.output_dir = str(tmp_path)
    hydra_dir = tmp_path.joinpath(".hydra")
    hydra_dir.mkdir(parents=True)
    hydra_dir.joinpath("config.yaml").write_text("some: config")
    mocker.patch("hydraflow.core.context.HydraConfig.get", return_value=hc)
    return hc


def test_log_run(hc: HydraConf) -> None:
    with mlflow.start_run() as run, log_run(run):
        Path(hc.runtime.output_dir).joinpath("output.log").write_text("output data")

    artifact_dir = get_artifact_dir(run)
    assert artifact_dir.joinpath(".hydra/config.yaml").read_text() == "some: config"
    assert artifact_dir.joinpath("output.log").read_text() == "output data"


def test_log_run_error() -> None:
    msg = "ValueError: Simulated error"
    with pytest.raises(ValueError, match=msg), mlflow.start_run() as run, log_run(run):
        raise ValueError(msg)


@pytest.mark.parametrize(("chdir", "expected"), [(True, True), (False, False)])
def test_start_run(chdir: bool, expected: bool) -> None:
    with start_run(chdir=chdir) as run:
        assert run is not None
        assert (Path.cwd() == get_artifact_dir(run)) == expected
