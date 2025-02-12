"""main decorator."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

import hydra
from hydra.core.config_store import ConfigStore
from mlflow.entities import RunStatus

import hydraflow

if TYPE_CHECKING:
    from collections.abc import Callable

    from mlflow.entities import Run

FINISHED = RunStatus.to_string(RunStatus.FINISHED)


def main(
    node: Any,
    config_name: str = "config",
    *,
    chdir: bool = False,
    force_new_run: bool = False,
    skip_finished: bool = True,
):
    """Main decorator."""

    def decorator(app: Callable[[Run, Any], None]) -> Callable[[], None]:
        ConfigStore.instance().store(name=config_name, node=node)

        @wraps(app)
        @hydra.main(version_base=None, config_name=config_name)
        def inner_app(cfg: object) -> None:
            hydraflow.set_experiment()

            if force_new_run:
                run = None
            else:
                rc = hydraflow.search_runs()
                run = rc.try_get(cfg, override=True)

                if skip_finished and run and run.info.status == FINISHED:
                    return

            with hydraflow.start_run(cfg, run=run, chdir=chdir) as run:
                app(run, cfg)

        return inner_app

    return decorator
