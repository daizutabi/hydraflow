"""Hydraflow jobs IO."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

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

    return OmegaConf.merge(schema, cfg)  # type: ignore


def get_job(name: str) -> Job:
    """Get a job from the config."""
    cfg = load_config()
    job = cfg.jobs[name]

    if not job.name:
        job.name = name

    return job
