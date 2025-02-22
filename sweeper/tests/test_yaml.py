from pathlib import Path

import pytest
from hydra.test_utils.test_utils import TSweepRunner
from omegaconf import OmegaConf


@pytest.fixture
def sweeper(hydra_sweep_runner: TSweepRunner):
    def sweeper(arguments: list[str], config_name: str):
        return hydra_sweep_runner(
            calling_file=None,
            calling_module="hydra.test_utils.a_module",
            config_path=f"{Path(__file__).parent}/configs",
            config_name=config_name,
            task_function=None,
            overrides=["hydra/sweeper=ext", "hydra/launcher=basic", *arguments],
        )

    return sweeper


def test_basic_int(sweeper) -> None:
    with sweeper(["x=1:2:5"], "basic.yaml") as sweep:
        assert sweep.returns is not None
        job_ret = sweep.returns[0]
        assert len(job_ret) == 3
        assert job_ret[0].overrides == ["x=1"]
        assert job_ret[0].cfg == {"x": 1, "y": 1e-2, "z": "abc"}
        assert job_ret[1].overrides == ["x=3"]
        assert job_ret[1].cfg == {"x": 3, "y": 1e-2, "z": "abc"}
        assert job_ret[2].overrides == ["x=5"]
        assert job_ret[2].cfg == {"x": 5, "y": 1e-2, "z": "abc"}


def test_basic_float(sweeper) -> None:
    with sweeper(["y=0:2:4:m"], "basic.yaml") as sweep:
        assert sweep.returns is not None
        job_ret = sweep.returns[0]
        assert len(job_ret) == 3
        assert job_ret[0].overrides == ["y=0"]
        assert job_ret[0].cfg == {"x": 1, "y": 0, "z": "abc"}
        assert job_ret[1].overrides == ["y=0.002"]
        assert job_ret[1].cfg == {"x": 1, "y": 2e-3, "z": "abc"}
        assert job_ret[2].overrides == ["y=0.004"]
        assert job_ret[2].cfg == {"x": 1, "y": 4e-3, "z": "abc"}


def test_basic_str(sweeper) -> None:
    with sweeper(["z=1:3"], "basic.yaml") as sweep:
        assert sweep.returns is not None
        job_ret = sweep.returns[0]
        assert len(job_ret) == 1
        assert job_ret[0].overrides == ["z=1\\:3"]
        assert job_ret[0].cfg == {"x": 1, "y": 1e-2, "z": "1:3"}


def test_nest_int(sweeper) -> None:
    with sweeper(["a.b=1:2:5"], "nest.yaml") as sweep:
        assert sweep.returns is not None
        job_ret = sweep.returns[0]
        assert len(job_ret) == 3
        assert job_ret[0].overrides == ["a.b=1"]
        assert job_ret[0].cfg == {"a": {"b": 1, "c": 0.2}}
        assert job_ret[1].overrides == ["a.b=3"]
        assert job_ret[1].cfg == {"a": {"b": 3, "c": 0.2}}
        assert job_ret[2].overrides == ["a.b=5"]
        assert job_ret[2].cfg == {"a": {"b": 5, "c": 0.2}}


def test_invalid_config():
    from hydra_plugins.hydra_ext_sweeper.ext_sweeper import ExtSweeper

    assert not ExtSweeper.is_number(OmegaConf.create({"a": {"b": 1}}), "a.c")
