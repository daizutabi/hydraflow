# Hydraflow

[![PyPI Version][pypi-v-image]][pypi-v-link]
[![Python Version][python-v-image]][python-v-link]
[![Build Status][GHAction-image]][GHAction-link]
[![Coverage Status][codecov-image]][codecov-link]

<!-- Badges -->
[pypi-v-image]: https://img.shields.io/pypi/v/hydraflow.svg
[pypi-v-link]: https://pypi.org/project/hydraflow/
[python-v-image]: https://img.shields.io/pypi/pyversions/hydraflow.svg
[python-v-link]: https://pypi.org/project/hydraflow
[GHAction-image]: https://github.com/daizutabi/hydraflow/actions/workflows/ci.yaml/badge.svg?branch=main&event=push
[GHAction-link]: https://github.com/daizutabi/hydraflow/actions?query=event%3Apush+branch%3Amain
[codecov-image]: https://codecov.io/github/daizutabi/hydraflow/coverage.svg?branch=main
[codecov-link]: https://codecov.io/github/daizutabi/hydraflow?branch=main

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

## Installation

You can install Hydraflow via pip:

```bash
pip install hydraflow
```

## Getting Started

Here is a simple example to get you started with Hydraflow:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import hydraflow

if TYPE_CHECKING:
    from mlflow.entities import Run


@dataclass
class Config:
    count: int = 1
    name: str = "a"


@hydraflow.main(Config)
def app(run: Run, cfg: Config):
    """Your app code here."""


if __name__ == "__main__":
    app()
```
