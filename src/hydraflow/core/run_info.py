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
from functools import cached_property
from typing import TYPE_CHECKING

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
        return get_job_name(self.run_dir)

    def to_dict(self) -> dict[str, Any]:
        """Convert the RunInfo to a dictionary."""
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir.as_posix(),
            "job_name": self.job_name,
        }


def get_job_name(run_dir: Path) -> str:
    """Extract the Hydra job name from the Hydra configuration file.

    Return an empty string if the job name cannot be extracted from the
    Hydra configuration file (e.g., if the file does not exist or does not
    contain the expected format).

    Args:
        run_dir (Path): The directory where the run artifacts are stored.

    Returns:
        str: The Hydra job name, which was used as the MLflow Experiment name.

    """
    hydra_file = run_dir / "artifacts/.hydra/hydra.yaml"

    if not hydra_file.exists():
        return ""

    text = hydra_file.read_text()
    if "  job:\n    name: " in text:
        return text.split("  job:\n    name: ")[1].split("\n")[0]

    return ""
