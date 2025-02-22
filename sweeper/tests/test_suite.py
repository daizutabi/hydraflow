import pytest
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.test_utils.launcher_common_tests import (
    BatchedSweeperTestSuite,
    IntegrationTestSuite,
    LauncherTestSuite,
)
from hydra.test_utils.test_utils import TSweepRunner

from hydra_plugins.hydra_ext_sweeper.ext_sweeper import ExtSweeper


def test_discovery() -> None:
    plugin_names = [x.__name__ for x in Plugins.instance().discover(Sweeper)]
    assert ExtSweeper.__name__ in plugin_names


def test_launched_jobs(hydra_sweep_runner: TSweepRunner) -> None:
    sweep = hydra_sweep_runner(
        calling_file=None,
        calling_module="hydra.test_utils.a_module",
        config_path="configs",
        config_name="compose.yaml",
        task_function=None,
        overrides=["hydra/sweeper=ext", "hydra/launcher=basic", "foo=1,2"],
    )
    with sweep:
        assert sweep.returns is not None
        job_ret = sweep.returns[0]
        assert len(job_ret) == 2
        assert job_ret[0].overrides == ["foo=1"]
        assert job_ret[0].cfg == {"foo": 1, "bar": 100}
        assert job_ret[1].overrides == ["foo=2"]
        assert job_ret[1].cfg == {"foo": 2, "bar": 100}


# Run launcher test suite with the basic launcher and this sweeper
@pytest.mark.parametrize(
    ("launcher_name", "overrides"),
    [("basic", ["hydra/sweeper=ext"])],
)
class TestExampleSweeper(LauncherTestSuite):
    pass


# Many sweepers are batching jobs in groups.
# This test suite verifies that the spawned jobs are not overstepping
# the directories of one another.
@pytest.mark.parametrize(
    ("launcher_name", "overrides"),
    # This will cause the sweeper to split batches to at most 2 jobs each, which is what
    # the tests in BatchedSweeperTestSuite are expecting.
    [("basic", ["hydra/sweeper=ext", "hydra.sweeper.max_batch_size=2"])],
)
class TestExperimentSweeperWithBatching(BatchedSweeperTestSuite):
    pass


# Run integration test suite with the basic launcher and this sweeper
@pytest.mark.parametrize(
    ("task_launcher_cfg", "extra_flags"),
    [({}, ["-m", "hydra/sweeper=ext", "hydra/launcher=basic"])],
)
class TestExperimentSweeperIntegration(IntegrationTestSuite):
    pass
