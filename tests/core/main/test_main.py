from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from hydra.conf import HydraConf
from hydra.types import RunMode

from hydraflow.core.main import equals, get_run_id, set_experiment

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.mark.parametrize("mode", [RunMode.RUN, RunMode.MULTIRUN])
def test_set_experiment(
    mode: RunMode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    hc = HydraConf(mode=mode)
    hc.sweep.dir = str(tmp_path.joinpath("sweeps"))
    hc.job.name = "my_experiment"
    db = str(Path("mlflow.db").absolute())
    e = set_experiment(hc, tracking_uri=f"sqlite:///{db}")
    assert e.name == "my_experiment"  # pyright: ignore[reportUnknownMemberType]
    assert Path("mlflow.db").is_file()
    if mode == RunMode.MULTIRUN:
        assert Path(".hydraflow.lock").is_file()
    else:
        assert not Path(".hydraflow.lock").exists()


@pytest.mark.parametrize(("mocked", "expected"), [(True, "run_123"), (False, None)])
def test_get_run_id(
    mocked: bool,
    expected: str | None,
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    mocker.patch("hydraflow.core.main.equals", return_value=mocked)
    tmp_path.joinpath("run_123").mkdir()
    run_id = get_run_id(str(tmp_path), {"some": "config"}, None)
    assert run_id == expected


def test_get_run_id_invalid_uri():
    assert get_run_id("invalid", None, None) is None


def test_get_run_id_empty_dir(tmp_path: Path):
    assert get_run_id(str(tmp_path), None, None) is None


def test_equals_config():
    assert equals(Path("a/b/c"), {"a": 1}, None) is False


def test_equals_overrides():
    assert equals(Path("a/b/c"), {"a": 1}, ["a=1"]) is False
