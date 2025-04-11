# Core Concepts

This page introduces the fundamental concepts of HydraFlow that form the
foundation of the framework.

## Run

A `Run` represents a single execution of a machine learning experiment in
HydraFlow. It is important to note that HydraFlow's `Run` class is distinct
from MLflow's `Run` class (`mlflow.entities.Run`). HydraFlow's `Run` is
designed with a focus on Hydra integration and configuration management.

HydraFlow's `Run` provides:

- Access to experiment configurations used during the run
- Methods for loading and analyzing experiment results
- Support for custom implementations through the factory pattern
- Type-safe access to configuration values

Example usage:

```python
from hydraflow import Run

# Load an existing run
run = Run.load("path/to/run")

# Access configuration, implementation attributes, or run info
learning_rate = run.get("learning_rate")  # From config
model_type = run.get("model.type")        # Access nested config with dot notation
accuracy = run.get("accuracy")            # Could be from implementation
run_dir = run.get("run_dir")              # From run info

# For MLflow specific data, you can use the underlying MLflow client
import mlflow
mlflow_run = mlflow.get_run(run_id=run.info.run_id)
metrics = mlflow_run.data.metrics
params = mlflow_run.data.params
```

The `Run` class serves as the primary interface for interacting with
individual experiment runs, allowing you to analyze results and compare
different approaches.

## RunCollection

A `RunCollection` is a collection of `Run` instances that provides tools
for analyzing and comparing multiple experiments. Key features include:

- Filtering runs based on configuration parameters
- Grouping runs by common attributes
- Aggregating data across runs
- Converting to DataFrames for analysis

Example usage:

```python
from hydraflow import Run

# Load multiple runs
runs = Run.load(["path/to/run1", "path/to/run2", "path/to/run3"])

# Filter runs by parameter value
filtered_runs = runs.filter(model_type="lstm")

# Group runs by a parameter
grouped_runs = runs.group_by("dataset.name")

# Convert to DataFrame for analysis
df = runs.to_frame("run_id", "learning_rate", "batch_size")
```

The `RunCollection` class enables comparative analysis across multiple
experiments, making it easy to identify trends and optimal configurations.

## Configuration Management

HydraFlow uses a hierarchical configuration system based on OmegaConf and
Hydra. This provides:

- Typed configuration with schema validation
- Configuration composition from multiple sources
- Dynamic configuration resolution
- Command-line overrides

Using configuration with HydraFlow:

```python
from dataclasses import dataclass
import hydraflow

@dataclass
class Config:
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10

@hydraflow.main(Config)
def train(run, cfg: Config) -> None:
    # Use configuration
    print(f"Training with lr={cfg.learning_rate}, batch_size={cfg.batch_size}")

    # Log metrics with MLflow
    mlflow.log_metric("accuracy", 0.95)
```

## Experiment Tracking

HydraFlow seamlessly integrates with MLflow, allowing you to use standard
MLflow functionality within your `hydraflow.main` decorated functions:

```python
from hydraflow import hydraflow
import mlflow

@hydraflow.main(Config)
def train(run, cfg: Config) -> None:
    # Your training code

    # Use standard MLflow APIs for tracking
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("model.pt")
```

## Summary

These core concepts work together to provide a comprehensive framework for
managing machine learning experiments:

- `Run` represents individual experiment runs with a focus on Hydra configuration
- `RunCollection` enables comparative analysis across multiple runs
- Configuration management with Hydra ensures reproducibility
- Experiment tracking with MLflow records metrics and artifacts

Understanding these fundamental concepts will help you leverage the full
power of HydraFlow for your machine learning projects.