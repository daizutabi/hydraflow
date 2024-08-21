from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import hydra
from hydra.main import _UNSPECIFIED_
from hydra.types import TaskFunction
from omegaconf import DictConfig


def main(
    config_path: str | None = _UNSPECIFIED_,
    config_name: str | None = None,
    version_base: str | None = _UNSPECIFIED_,
) -> Callable[[TaskFunction], Any]:
    _main_decorator = hydra.main(
        config_path=config_path,
        config_name=config_name,
        version_base=version_base,
    )

    def main_decorator(task_function: TaskFunction) -> Callable[[], None]:
        print(inspect.getsourcefile(task_function))
        print(inspect.getsource(task_function))
        print(task_function.__name__)
        _decorated_main = _main_decorator(task_function)

        def decorated_main(cfg_passthrough: DictConfig | None = None) -> Any:
            print(cfg_passthrough)
            return _decorated_main(cfg_passthrough)

        return decorated_main

    return main_decorator
