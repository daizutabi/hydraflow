# Core Concepts

This page introduces the fundamental concepts of HydraFlow that
form the foundation of the framework.

## Design Principles

HydraFlow is built on the following design principles:

1. **Type Safety** - Utilizing Python dataclasses for configuration
    type checking and IDE support
2. **Reproducibility** - Automatically tracking all experiment configurations
    for fully reproducible experiments
3. **Workflow Integration** - Creating a cohesive workflow by integrating
    Hydra's configuration management with MLflow's experiment tracking
4. **Analysis Capabilities** - Providing powerful APIs for easily
    analyzing experiment results

## Key Components

HydraFlow consists of the following key components:

### Configuration Management

HydraFlow uses a hierarchical configuration system based on
OmegaConf and Hydra. This provides:

- Type-safe configuration using Python dataclasses
- Schema validation to ensure configuration correctness
- Configuration composition from multiple sources
- Command-line overrides

Example configuration:

```python
from dataclasses import dataclass

@dataclass
class Config:
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
```

This configuration class defines the structure and default values
for your experiment, enabling type checking and auto-completion.

### Main Decorator

The [`@hydraflow.main`][hydraflow.main] decorator defines the entry
point for a HydraFlow application:

```python
import hydraflow
from mlflow.entities import Run

@hydraflow.main(Config, tracking_uri="sqlite:///mlflow.db")
def train(run: Run, cfg: Config) -> None:
    # Your experiment code
    print(f"Training with lr={cfg.learning_rate}, batch_size={cfg.batch_size}")

    # Log metrics
    hydraflow.log_metric("accuracy", 0.95)
```

This decorator provides:

- Automatic registration of your config class with Hydra's `ConfigStore`
- Automatic setup of an MLflow experiment (including tracking URI)
- Storage of Hydra configurations and logs as MLflow artifacts
- Support for type-safe APIs and IDE integration

### Workflow Automation

HydraFlow allows you to automate experiment workflows using a
YAML-based job definition system:

```yaml
jobs:
  train_models:
    run: python train.py
    sets:
      - each: model=small,medium,large
        all: learning_rate=0.001,0.01,0.1
```

This enables:

- Defining reusable experiment workflows
- Efficient configuration of parameter sweeps
- Organization of complex experiment campaigns

You can also define more complex parameter spaces using extended sweep syntax:

```bash
# Ranges (start:end:step)
python train.py -m "learning_rate=0.01:0.03:0.01"

# SI prefixes
python train.py -m "batch_size=1k,2k,4k"
# 1000, 2000, 4000

# Grid within a single parameter
python train.py -m "model=(small,large)_(v1,v2)"
# small_v1, small_v2, large_v1, large_v2
```

### Analysis Tools

After running experiments, HydraFlow provides powerful tools for accessing
and analyzing results. These tools help you track, compare, and derive
insights from your experiments.

#### Working with Individual Runs

For individual experiment analysis, HydraFlow provides the `Run` class,
which represents a single experiment run:

```python
from hydraflow import Run

# Load an existing run
run = Run.load("path/to/run")

# Access configuration values
learning_rate = run.get("learning_rate")
```

The `Run` class provides:

- Access to experiment configurations used during the run
- Methods for loading and analyzing experiment results
- Support for custom implementations through the factory pattern
- Type-safe access to configuration values

You can use type parameters for more powerful IDE support:

```python
from dataclasses import dataclass
from hydraflow import Run

@dataclass
class MyConfig:
    learning_rate: float
    batch_size: int

# Load a Run with type information
run = Run[MyConfig].load("path/to/run")
print(run.cfg.learning_rate)  # IDE auto-completion works
```

#### Comparing Multiple Runs

For comparing multiple runs, HydraFlow offers the `RunCollection` class,
which enables efficient analysis across runs:

```python
# Load multiple runs
runs = Run.load(["path/to/run1", "path/to/run2", "path/to/run3"])

# Filter runs by parameter value
filtered_runs = runs.filter(model_type="lstm")

# Group runs by a parameter
grouped_runs = runs.group_by("dataset_name")

# Convert to DataFrame for analysis
df = runs.to_frame("learning_rate", "batch_size", "accuracy")
```

Key features of experiment comparison:

- Filtering runs based on configuration parameters
- Grouping runs by common attributes
- Aggregating data across runs
- Converting to Polars DataFrames for advanced analysis

## Summary

These core concepts work together to provide a comprehensive framework
for managing machine learning experiments:

1. **Configuration Management** - Type-safe configuration with Python dataclasses
2. **Main Decorator** - The entry point that integrates Hydra and MLflow
3. **Workflow Automation** - Reusable experiment definitions and advanced parameter sweeps
4. **Analysis Tools** - Access, filter, and analyze experiment results

Understanding these fundamental concepts will help you leverage the full power
of HydraFlow for your machine learning projects.
