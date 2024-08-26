# Hydraflow Documentation

Welcome to the Hydraflow documentation. This guide will help you understand how to use Hydraflow to manage and track your machine learning experiments by integrating Hydra and MLflow.

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

To get started with Hydraflow, follow these steps:

1. **Install Hydraflow**: Follow the installation instructions above.

2. **Create a Hydra configuration file**: Define your experiment configuration using Hydra's syntax.

3. **Create a Python script to run your experiment**:

```python
import hydra
import mlflow
from omegaconf import DictConfig

@hydra.main(config_name="config")
def my_app(cfg: DictConfig) -> None:
    with mlflow.start_run(), hydraflow.log_run(cfg):
        # Your training code here
        pass

if __name__ == "__main__":
    my_app()
```
