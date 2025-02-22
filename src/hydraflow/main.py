"""main decorator."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from mlflow.entities import RunStatus
from omegaconf import OmegaConf

import hydraflow

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mlflow.entities import Run

FINISHED = RunStatus.to_string(RunStatus.FINISHED)


def main(
    node: Any,
    config_name: str = "config",
    *,
    chdir: bool = False,
    force_new_run: bool = False,
    use_overrides: bool = False,
    rerun_finished: bool = False,
):
    """Main decorator."""

    def decorator(app: Callable[[Run, Any], None]) -> Callable[[], None]:
        ConfigStore.instance().store(name=config_name, node=node)

        @wraps(app)
        @hydra.main(version_base=None, config_name=config_name)
        def inner_app(cfg: object) -> None:
            hc = HydraConfig.get()
            experiment = mlflow.set_experiment(hc.job.name)

            if force_new_run:
                run_id = None
            else:
                run_id = get_run_id(cfg, experiment.name, overrides=use_overrides)

                if run_id and not rerun_finished:
                    run = mlflow.get_run(run_id)
                    if run.info.status == FINISHED:
                        return

            with hydraflow.start_run(cfg, run_id=run_id, chdir=chdir) as run:
                app(run, cfg)

        return inner_app

    return decorator


def get_run_id(
    cfg: object,
    experiment_name: str,
    *,
    overrides: bool = False,
) -> str | None:
    """Try to get the run ID for the given configuration.

    If the run is not found, the function will return None.

    Args:
        cfg (object): The configuration object.
        experiment_name (str): The name of the experiment.
        overrides (bool): Whether to use overrides.

    Returns:
        The run ID for the given configuration. Returns None if
        no run ID is found.

    """
    artifact_dirs = hydraflow.list_run_paths(experiment_name, "artifacts")
    for artifact_dir in artifact_dirs:
        if _equals(cfg, artifact_dir, overrides=overrides):
            return artifact_dir.parent.name

    return None


def _equals(cfg: object, artifact_dir: Path, *, overrides: bool = False) -> bool:
    if overrides:
        return _equals_overrides(artifact_dir)

    return _equals_config(cfg, artifact_dir)


def _equals_config(cfg: object, artifact_dir: Path) -> bool:
    path = artifact_dir / ".hydra/config.yaml"
    if not path.exists():
        return False

    return cfg == OmegaConf.load(path)


def _equals_overrides(artifact_dir: Path) -> bool:
    path = artifact_dir / ".hydra/overrides.yaml"
    if not path.exists():
        return False

    overrides = HydraConfig.get().overrides.task
    return overrides == OmegaConf.load(path)
