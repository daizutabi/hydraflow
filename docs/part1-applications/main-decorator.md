# Main Decorator

The [`hydraflow.main`][hydraflow.main] decorator is the central component
for creating HydraFlow applications. It bridges Hydra's configuration
management with MLflow's experiment tracking, automatically setting up
the experiment environment.

## Basic Usage

Here's how to use the main decorator in its simplest form:

```python
from dataclasses import dataclass
import hydraflow

@dataclass
class Config:
    learning_rate: float = 0.01
    batch_size: int = 32

@hydraflow.main(Config)
def train(run, cfg: Config) -> None:
    print(f"Training with learning_rate={cfg.learning_rate}")
    # Your training code here

if __name__ == "__main__":
    train()
```

## Function Signature

The function decorated with [`@hydraflow.main`][hydraflow.main] must accept
two parameters:

1. `run`: The current MLflow run object, which can be used to access run
   information and log additional metrics or artifacts.

2. `cfg`: The configuration object containing all parameters, populated from
   Hydra's configuration system and command-line overrides.

## Type Annotations

The `cfg` parameter should be annotated with your configuration class for type
checking and IDE auto-completion. This is particularly useful when working
with complex configurations:

```python
@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int

@dataclass
class DataConfig:
    path: str
    validation_split: float

@dataclass
class Config:
    training: TrainingConfig
    data: DataConfig
    seed: int = 42

@hydraflow.main(Config)
def train(run, cfg: Config) -> None:
    # Type-checked access to nested configuration
    lr = cfg.training.learning_rate
    data_path = cfg.data.path

    # Your training code here
```

## Available HydraFlow Functions

Within a function decorated with [`@hydraflow.main`][hydraflow.main], you have
access to various HydraFlow utilities for logging:

```python
@hydraflow.main(Config)
def train(run, cfg: Config) -> None:
    # Log metrics
    hydraflow.log_metric("accuracy", 0.95)

    # Log a set of metrics
    hydraflow.log_metrics({
        "precision": 0.92,
        "recall": 0.89,
        "f1_score": 0.90
    })

    # Log artifacts
    hydraflow.log_artifact("model.pkl", "Trained model")

    # Log parameters not included in the config
    hydraflow.log_param("custom_param", "value")
```

## Advanced Features

### Customizing the Run Name

You can customize the MLflow run name by providing a `run_name` parameter:

```python
@hydraflow.main(Config, run_name="custom_experiment")
def train(run, cfg: Config) -> None:
    # Training code
```

### Tags and Custom Logging

Add custom tags to the run for better organization:

```python
@hydraflow.main(Config)
def train(run, cfg: Config) -> None:
    # Add custom tags
    hydraflow.set_tag("model_version", "v2.0")
    hydraflow.set_tag("data_version", "2023-01-15")

    # Your training code here
```

### Accessing Run Information

The `run` parameter provides direct access to run information:

```python
@hydraflow.main(Config)
def train(run, cfg: Config) -> None:
    # Access run information
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
    print(f"Status: {run.info.status}")

    # Your training code here
```

## Best Practices

1. **Keep Configuration Classes Focused**: Break down complex configurations
   into logical components using nested dataclasses.

2. **Use Type Annotations**: Always annotate your function parameters for
   better IDE support and type checking.

3. **Log Important Information**: Log all relevant metrics, parameters, and
   artifacts to ensure reproducibility.

4. **Handle Errors Gracefully**: Implement proper error handling inside your
   main function to avoid losing experiment data.

## Summary

The [`hydraflow.main`][hydraflow.main] decorator simplifies the integration of
Hydra and MLflow, handling configuration management and experiment tracking
automatically. This allows you to focus on your experiment implementation
while ensuring that all relevant information is properly tracked and organized.