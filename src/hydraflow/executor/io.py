"""Hydraflow jobs IO."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

from .conf import HydraflowConf

if TYPE_CHECKING:
    from .conf import Job


def load_config(config_file: str | Path = "hydraflow.yaml") -> HydraflowConf:
    """Load the hydraflow config."""
    schema = OmegaConf.structured(HydraflowConf)

    config_file = Path(config_file)

    if not config_file.exists():
        return schema

    cfg = OmegaConf.load(config_file)

    if not isinstance(cfg, DictConfig):
        return schema

    return OmegaConf.merge(schema, cfg)  # pyright: ignore[reportReturnType]


def get_job(name: str, config_file: str | Path = "hydraflow.yaml") -> Job:
    """Get a job from the config."""
    cfg = load_config(config_file)
    job = cfg.jobs[name]

    if not job.name:
        job.name = name

    return job
