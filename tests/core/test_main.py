from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import pytest
from hydra.conf import HydraConf
from hydra.types import RunMode
from omegaconf import OmegaConf

from hydraflow.core.main import equals, get_run_id, main, set_experiment

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def cfg() -> dict[str, int]:
    return {"a": 1, "b": 2}


@pytest.fixture
def hc(cfg: dict[str, int], mocker: MockerFixture, tmp_path: Path) -> HydraConf:
    hc = HydraConf()
    hc.runtime.output_dir = str(tmp_path)
    hc.job.name = "test_job"

    hydra_dir = tmp_path.joinpath(".hydra")
    hydra_dir.mkdir(parents=True)
    cfg_path = hydra_dir.joinpath("config.yaml")
    OmegaConf.save(cfg, cfg_path)
    mocker.patch("hydraflow.core.main.HydraConfig.get", return_value=hc)
    return hc


@pytest.fixture(params=["mlruns", "mlflow.db"])
def uri(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    hc: HydraConf,  # pyright: ignore[reportUnusedParameter]
) -> str | Path:
    monkeypatch.chdir(tmp_path)
    assert isinstance(request.param, str)

    mocker.patch("sys.argv")

    if "." in request.param:
        db = (tmp_path / request.param).as_posix()
        return f"sqlite:///{db}"

    return tmp_path / request.param


@dataclass
class Config:
    a: int = 1
    b: int = 2


def test_main(uri: str | Path, mocker: MockerFixture) -> None:
    app = mocker.MagicMock()
    deco = main(Config, tracking_uri=uri)(app)
    deco()
    deco()
    app.assert_called_once()
    runs = mlflow.search_runs(experiment_names=["test_job"])
    assert len(runs) == 1


def test_main_rerun_finished(uri: str | Path, mocker: MockerFixture) -> None:
    app = mocker.MagicMock()
    deco = main(Config, rerun_finished=True, tracking_uri=uri)(app)
    deco()
    deco()
    assert app.call_count == 2
    runs = mlflow.search_runs(experiment_names=["test_job"])
    assert len(runs) == 1


def test_main_force_new_run(uri: str | Path, mocker: MockerFixture) -> None:
    app = mocker.MagicMock()
    deco = main(Config, force_new_run=True, tracking_uri=uri)(app)
    deco()
    deco()
    assert app.call_count == 2
    runs = mlflow.search_runs(experiment_names=["test_job"])
    assert len(runs) == 2


def test_main_update(uri: str | Path, hc: HydraConf, mocker: MockerFixture) -> None:
    def update(_cfg: Config) -> Config:
        return Config(100, 200)

    app = mocker.MagicMock()
    deco = main(Config, update=update, tracking_uri=uri)(app)
    deco()
    text = Path(hc.runtime.output_dir).joinpath(".hydra/config.yaml").read_text()
    assert text == "a: 100\nb: 200\n"


def test_main_dry_run(uri: str | Path, mocker: MockerFixture) -> None:
    app = mocker.MagicMock()
    deco = main(Config, dry_run=True, tracking_uri=uri)(app)
    deco()
    app.assert_not_called()


def test_main_dry_run_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["python", "app.py", "--dry-run"])
    main(Config)
    assert "--dry-run" not in sys.argv


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


@pytest.mark.parametrize(("mocked", "expected"), [(True, "run_123"), (False, None)])
def test_get_run_id(
    *,
    mocked: bool,
    expected: str | None,
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    mocker.patch("hydraflow.core.main.equals", return_value=mocked)
    tmp_path.joinpath("run_123").mkdir()
    run_id = get_run_id(str(tmp_path), {"some": "config"}, None)
    assert run_id == expected


def test_get_run_id_invalid_uri() -> None:
    assert get_run_id("invalid", None, None) is None


def test_get_run_id_empty_dir(tmp_path: Path) -> None:
    assert get_run_id(str(tmp_path), None, None) is None


@pytest.mark.parametrize(("a", "expected"), [(1, True)])
def test_equals_config(a: int, *, expected: bool, tmp_path: Path) -> None:
    hydra_dir = tmp_path.joinpath("artifacts/.hydra")
    hydra_dir.mkdir(parents=True)
    hydra_dir.joinpath("config.yaml").write_text("a: 1\n")
    assert equals(tmp_path, {"a": a}, None) is expected


def test_equals_config_invalid_path() -> None:
    assert equals(Path("a/b/c"), {"a": 1}, None) is False


@pytest.mark.parametrize(("a", "expected"), [(1, True)])
def test_equals_overdies(a: int, *, expected: bool, tmp_path: Path) -> None:
    hydra_dir = tmp_path.joinpath("artifacts/.hydra")
    hydra_dir.mkdir(parents=True)
    hydra_dir.joinpath("overrides.yaml").write_text("a=1\n")
    assert equals(tmp_path, {"a": a}, [f"a={a}"]) is expected


def test_equals_overrides_invalid_path() -> None:
    assert equals(Path("a/b/c"), {"a": 1}, ["a=1"]) is False
