"""RunInfo module for HydraFlow.

This module provides the RunInfo class, which represents a
MLflow Run in HydraFlow. RunInfo contains information about a run,
such as the run directory, run ID, and job name.
The job name is extracted from the Hydra configuration file and
represents the MLflow Experiment name that was used when the run
was created.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


@dataclass
class RunInfo:
    """Information about a MLflow Run in HydraFlow.

    This class represents a MLflow Run and contains information
    such as the run directory, run ID, and job name.
    The job name is extracted from the Hydra configuration file
    and represents the MLflow Experiment name that was used when
    the run was created.

    """

    run_dir: Path
    """The MLflow Run directory, which contains metrics, parameters, and artifacts."""

    @cached_property
    def run_id(self) -> str:
        """The MLflow run ID, which is the name of the run directory."""
        return self.run_dir.name

    @cached_property
    def job_name(self) -> str:
        """The Hydra job name, which was used as the MLflow Experiment name.

        An empty string if the job name cannot be extracted from the
        Hydra configuration file (e.g., if the file does not exist or does not
        contain the expected format).
        """
        return get_job_name(self.run_dir.parent)

    def to_dict(self) -> dict[str, Any]:
        """Convert the RunInfo to a dictionary."""
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir.as_posix(),
            "job_name": self.job_name,
        }


@cache
def get_job_name(experiment_dir: Path) -> str:
    """Get the job name from an experiment directory.

    Extracts the job name from the meta.yaml file. Returns an empty string
    if the file does not exist or if the job name cannot be found.

    Args:
        experiment_dir: Path to the experiment directory containing the meta.yaml file

    Returns:
        The job name as a string, or an empty string if the file does not exist

    """
    path = experiment_dir / "meta.yaml"
    if not path.exists():
        return ""

    meta = OmegaConf.load(experiment_dir / "meta.yaml")
    return OmegaConf.select(meta, "name")
