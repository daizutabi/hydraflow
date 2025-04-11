# Running Applications with HydraFlow

This section covers the fundamentals of defining and running HydraFlow
applications - the first step in your machine learning experiment workflow.

## Overview

HydraFlow applications combine Hydra's configuration management with
MLflow's experiment tracking capabilities. This allows you to:

- Define type-safe configurations using Python dataclasses
- Override configurations via command-line arguments
- Launch multiple runs with different parameter combinations
- Automatically track all configurations, metrics, and artifacts

## Key Components

The core elements of a HydraFlow application are:

1. **Configuration Class**: A Python dataclass that defines the structure
   and default values of your application's configuration.

2. **Main Function**: The entry point of your application, decorated with
   [`@hydraflow.main`][hydraflow.main] and accepting the configuration as
   an argument.

3. **Experiment Logic**: Your machine learning code that utilizes the
   configuration and logs results.

## Basic Application Structure

```python
from dataclasses import dataclass
from mlflow.entities import Run
import hydraflow

@dataclass
class Config:
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 10

@hydraflow.main(Config)
def app(run: Run, cfg: Config) -> None:
    # Your experiment code here
    print(f"Training with lr={cfg.learning_rate}, batch_size={cfg.batch_size}")

    # Log metrics and artifacts
    hydraflow.log_metric("accuracy", 0.95)
    hydraflow.log_artifact("model.pkl", "Model checkpoint")

if __name__ == "__main__":
    app()
```

## What's Next

In the following pages, we'll explore each aspect of HydraFlow applications
in detail:

- [Main Decorator](main-decorator.md): Learn how to use the
  [`@hydraflow.main`][hydraflow.main] decorator to create HydraFlow applications.

- [Configuration](configuration.md): Discover how to define, compose, and
  validate configurations using dataclasses and Hydra's powerful
  configuration system.

- [Execution](execution.md): Understand how to run applications, override
  configurations, and perform parameter sweeps for experimentation.

## Advanced Application Features

Once you've mastered the basics, you may want to explore HydraFlow's more advanced features:

- [Extended Sweep Syntax](../part3-advanced/sweep-syntax.md): Define complex parameter spaces using
  HydraFlow's powerful syntax for numerical ranges, combinations, and more.

- [Job Configuration](../part3-advanced/job-configuration.md): Create reusable job definitions for
  repeated experiment workflows using a declarative YAML format.