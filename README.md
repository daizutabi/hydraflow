# Hydraflow

[![PyPI Version][pypi-v-image]][pypi-v-link]
[![Build Status][GHAction-image]][GHAction-link]
[![Coverage Status][codecov-image]][codecov-link]
[![Documentation Status][docs-image]][docs-link]
[![Python Version][python-v-image]][python-v-link]

<!-- Badges -->
[pypi-v-image]: https://img.shields.io/pypi/v/hydraflow.svg
[pypi-v-link]: https://pypi.org/project/hydraflow/
[GHAction-image]: https://github.com/daizutabi/hydraflow/actions/workflows/ci.yaml/badge.svg?branch=main&event=push
[GHAction-link]: https://github.com/daizutabi/hydraflow/actions?query=event%3Apush+branch%3Amain
[codecov-image]: https://codecov.io/github/daizutabi/hydraflow/coverage.svg?branch=main
[codecov-link]: https://codecov.io/github/daizutabi/hydraflow?branch=main
[docs-image]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-link]: https://daizutabi.github.io/hydraflow/
[python-v-image]: https://img.shields.io/pypi/pyversions/hydraflow.svg
[python-v-link]: https://pypi.org/project/hydraflow

## Overview

Hydraflow is a library designed to seamlessly integrate
[Hydra](https://hydra.cc/) and [MLflow](https://mlflow.org/), making it easier to
manage and track machine learning experiments. By combining the flexibility of
Hydra's configuration management with the robust experiment tracking capabilities
of MLflow, Hydraflow provides a comprehensive solution for managing complex
machine learning workflows.

## Key Features

- **Configuration Management**: Utilize Hydra's advanced configuration management
  to handle complex parameter sweeps and experiment setups.
- **Experiment Tracking**: Leverage MLflow's tracking capabilities to log parameters,
  metrics, and artifacts for each run.
- **Artifact Management**: Automatically log and manage artifacts, such as model
  checkpoints and configuration files, with MLflow.
- **Seamless Integration**: Easily integrate Hydra and MLflow in your machine learning
  projects with minimal setup.
- **Rich CLI Interface**: Command-line tools for managing experiments and viewing results.
- **Cross-Platform Support**: Works consistently across different operating systems.

## Installation

You can install Hydraflow via pip:

```bash
pip install hydraflow
```

**Requirements:** Python 3.13+

## Quick Start

Here is a simple example to get you started with Hydraflow:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydraflow
import mlflow

if TYPE_CHECKING:
    from mlflow.entities import Run


@dataclass
class Config:
    """Configuration for the ML training experiment."""
    # Training hyperparameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10

    # Model architecture parameters
    hidden_size: int = 128
    dropout: float = 0.1

    # Dataset parameters
    train_size: float = 0.8
    random_seed: int = 42


@hydraflow.main(Config)
def app(run: Run, cfg: Config):
    """Train a model with the given configuration.

    This example demonstrates how to:

    1. Define a configuration using dataclasses
    2. Use Hydraflow to integrate with MLflow
    3. Track metrics and parameters automatically

    Args:
        run: MLflow run for the experiment corresponding to the Hydra app.
            This `Run` instance is automatically created by Hydraflow.
        cfg: Configuration for the experiment's run.
            This `Config` instance is originally defined by Hydra, and then
            automatically passed to the app by Hydraflow.
    """
    # Training loop
    for epoch in range(cfg.epochs):
        # Simulate training and validation
        train_loss = 1.0 / (epoch + 1)
        val_loss = 1.1 / (epoch + 1)

        # Log metrics to MLflow
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")


if __name__ == "__main__":
    app()
```

This example demonstrates:

- Configuration management with Hydra
- Automatic experiment tracking with MLflow
- Parameter logging and metric tracking
- Type-safe configuration with dataclasses

## Documentation

For detailed documentation, including advanced usage examples and API reference,
visit our [documentation site](https://daizutabi.github.io/hydraflow/).

## Contributing

We welcome contributions! Please see our [contributing guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.