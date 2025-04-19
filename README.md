# HydraFlow

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

HydraFlow seamlessly integrates [Hydra](https://hydra.cc/) and [MLflow](https://mlflow.org/) to streamline machine learning experiment workflows. By combining Hydra's powerful configuration management with MLflow's robust experiment tracking, HydraFlow provides a comprehensive solution for defining, executing, and analyzing machine learning experiments.

## Design Principles

HydraFlow is built on the following design principles:

1. **Type Safety** - Utilizing Python dataclasses for configuration type checking and IDE support
2. **Reproducibility** - Automatically tracking all experiment configurations for fully reproducible experiments
3. **Analysis Capabilities** - Providing powerful APIs for easily analyzing experiment results
4. **Workflow Integration** - Creating a cohesive workflow by integrating Hydra's configuration management with MLflow's experiment tracking

## Key Features

- **Type-safe Configuration Management** - Define experiment parameters using Python dataclasses with full IDE support and validation
- **Seamless Hydra-MLflow Integration** - Automatically register configurations with Hydra and track experiments with MLflow
- **Advanced Parameter Sweeps** - Define complex parameter spaces using extended sweep syntax for numerical ranges, combinations, and SI prefixes
- **Workflow Automation** - Create reusable experiment workflows with YAML-based job definitions
- **Powerful Analysis Tools** - Filter, group, and analyze experiment results with type-aware APIs
- **Custom Implementation Support** - Extend experiment analysis with domain-specific functionality

## Installation

```bash
pip install hydraflow
```

**Requirements:** Python 3.13+

## Quick Example

```python
from dataclasses import dataclass
from mlflow.entities import Run
import hydraflow

@dataclass
class Config:
    width: int = 1024
    height: int = 768

@hydraflow.main(Config)
def app(run: Run, cfg: Config) -> None:
    # Your experiment code here
    print(f"Running with width={cfg.width}, height={cfg.height}")

    # Log metrics
    hydraflow.log_metric("area", cfg.width * cfg.height)

if __name__ == "__main__":
    app()
```

Execute a parameter sweep with:

```bash
python app.py -m width=800,1200 height=600,900
```

## Core Components

HydraFlow consists of the following key components:

### Configuration Management

Define type-safe configurations using Python dataclasses:

```python
@dataclass
class Config:
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
```

### Main Decorator

The `@hydraflow.main` decorator integrates Hydra and MLflow:

```python
@hydraflow.main(Config)
def train(run: Run, cfg: Config) -> None:
    # Your experiment code
```

### Workflow Automation

Define reusable experiment workflows in YAML:

```yaml
jobs:
  train_models:
    run: python train.py
    sets:
      - each: model=small,medium,large
        all: learning_rate=0.001,0.01,0.1
```

### Analysis Tools

Analyze experiment results with powerful APIs:

```python
from hydraflow import Run

# Load runs
runs = Run.load(iter_run_dirs("mlruns"))

# Filter and analyze
best_runs = runs.filter(model_type="transformer").to_frame("learning_rate", "accuracy")
```

## Documentation

For detailed documentation, visit our [documentation site](https://daizutabi.github.io/hydraflow/):

- [Getting Started](https://daizutabi.github.io/hydraflow/getting-started/) - Installation and core concepts
- [Practical Tutorials](https://daizutabi.github.io/hydraflow/practical-tutorials/) - Learn through hands-on examples
- [User Guide](https://daizutabi.github.io/hydraflow/part1-applications/) - Detailed documentation of HydraFlow's capabilities
- [API Reference](https://daizutabi.github.io/hydraflow/api/hydraflow/) - Complete API documentation

## Contributing

We welcome contributions! Please see our [contributing guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
