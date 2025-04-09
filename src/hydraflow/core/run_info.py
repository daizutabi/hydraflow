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

        Raises:
            FileNotFoundError: If the Hydra configuration file does not exist.
            ValueError: If the job name cannot be extracted from the
                configuration file.

        """
        return get_job_name(self.run_dir)


def get_job_name(run_dir: Path) -> str:
    """Extract the Hydra job name from the Hydra configuration file.

    Args:
        run_dir (Path): The directory where the run artifacts are stored.

    Returns:
        str: The Hydra job name, which was used as the MLflow Experiment name.

    Raises:
        FileNotFoundError: If the Hydra configuration file does not exist.
        ValueError: If the job name cannot be extracted from the
            configuration file.

    """
    hydra_file = run_dir / "artifacts/.hydra/hydra.yaml"

    if not hydra_file.exists():
        msg = f"Hydra configuration file not found at {hydra_file}. "
        msg += "This is required by HydraFlow conventions."
        raise FileNotFoundError(msg)

    text = hydra_file.read_text()
    if "  job:\n    name: " in text:
        return text.split("  job:\n    name: ")[1].split("\n")[0]

    msg = f"Could not extract job name from {hydra_file}. "
    msg += "The file should contain a 'hydra.job.name' field."
    raise ValueError(msg)
