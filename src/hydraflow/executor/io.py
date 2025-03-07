"""Hydraflow jobs IO."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig, ListConfig, OmegaConf

from .conf import HydraflowConf

if TYPE_CHECKING:
    from .job import Job


def find_config_file() -> Path | None:
    """Find the hydraflow config file."""
    if Path("hydraflow.yaml").exists():
        return Path("hydraflow.yaml")

    if Path("hydraflow.yml").exists():
        return Path("hydraflow.yml")

    return None


def load_config() -> HydraflowConf:
    """Load the hydraflow config."""
    schema = OmegaConf.structured(HydraflowConf)

    path = find_config_file()

    if path is None:
        return schema

    cfg = OmegaConf.load(path)

    if not isinstance(cfg, DictConfig):
        return schema

    rename_with(cfg)

    return OmegaConf.merge(schema, cfg)  # type: ignore[return-value]


def rename_with(cfg: DictConfig) -> None:
    """Rename the `with` field to `with_`."""
    if "with" in cfg:
        cfg["with_"] = cfg.pop("with")

    for key in list(cfg.keys()):
        if isinstance(cfg[key], DictConfig):
            rename_with(cfg[key])
        elif isinstance(cfg[key], ListConfig):
            for item in cfg[key]:
                if isinstance(item, DictConfig):
                    rename_with(item)


def get_job(name: str) -> Job:
    """Get a job from the config."""
    cfg = load_config()
    job = cfg.jobs[name]

    if not job.name:
        job.name = name

    return job
