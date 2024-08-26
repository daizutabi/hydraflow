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
[GHAction-image]: https://github.com/daizutabi/hydraflow/actions/workflows/ci.yml/badge.svg?branch=main&event=push
[GHAction-link]: https://github.com/daizutabi/hydraflow/actions?query=event%3Apush+branch%3Amain
[codecov-image]: https://codecov.io/github/daizutabi/hydraflow/coverage.svg?branch=main
[codecov-link]: https://codecov.io/github/daizutabi/hydraflow?branch=main

## Overview

Hydraflow is a powerful library designed to seamlessly integrate [Hydra](https://hydra.cc/) and [MLflow](https://mlflow.org/), making it easier to manage and track machine learning experiments. By combining the flexibility of Hydra's configuration management with the robust experiment tracking capabilities of MLflow, Hydraflow provides a comprehensive solution for managing complex machine learning workflows.

## Key Features

- **Configuration Management**: Utilize Hydra's advanced configuration management to handle complex parameter sweeps and experiment setups.
- **Experiment Tracking**: Leverage MLflow's tracking capabilities to log parameters, metrics, and artifacts for each run.
- **Artifact Management**: Automatically log and manage artifacts, such as model checkpoints and configuration files, with MLflow.
- **Seamless Integration**: Easily integrate Hydra and MLflow in your machine learning projects with minimal setup.

## Installation

You can install Hydraflow via pip:

```bash
pip install hydraflow
```

## Getting Started

Here is a simple example to get you started with Hydraflow:

```python
import hydra
import hydraflow
import mlflow
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class MySQLConfig:
    host: str = "localhost"
    port: int = 3306

cs = ConfigStore.instance()
cs.store(name="config", node=MySQLConfig)

@hydra.main(version_base=None, config_name="config")
def my_app(cfg: MySQLConfig) -> None:
    # Set experiment by Hydra job name.
    hydraflow.set_experiment()

    # Automatically log params using Hydra config.
    with mlflow.start_run(), hydraflow.log_run(cfg) as info:
        # Your app code below.

        # `info.output_dir` is the Hydra output directory.
        # `info.artifact_dir` is the MLflow run directory.

        with hydraflow.chdir_artifact():
            # Current working directory is the MLflow artifact directory.
            # You can use the directory to save your model or
            # other artifacts.
            pass

        with hydraflow.watch():
            # Watch files in the MLflow artifact directory.
            # You can update metrics or log other artifacts
            # according to the watched files.
            pass



if __name__ == "__main__":
    my_app()
```
