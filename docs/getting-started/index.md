# Getting Started with HydraFlow

Welcome to HydraFlow, a framework designed to streamline machine learning
workflows by integrating Hydra's configuration management with MLflow's
experiment tracking capabilities.

## Overview

This section provides everything you need to begin using HydraFlow
effectively:

- [Installation](installation.md): Step-by-step instructions for installing
  HydraFlow and its dependencies
- [Core Concepts](concepts.md): An introduction to the fundamental concepts
  that underpin HydraFlow's design and functionality

## Why HydraFlow?

Managing machine learning experiments involves numerous challenges, including:

- **Configuration Management**: Tracking hyperparameters and settings across
  multiple experiment runs
- **Reproducibility**: Ensuring experiments can be reliably reproduced
- **Result Analysis**: Efficiently comparing and analyzing experiment outcomes
- **Workflow Automation**: Organizing and managing experiment workflows

HydraFlow addresses these challenges by providing:

1. **Type-safe Configuration**: Using Python's native dataclasses for
   robust configuration management
2. **Seamless Integration**: Bridging Hydra and MLflow to combine their
   respective strengths
3. **Analysis Tools**: Providing powerful APIs for filtering, grouping,
   and analyzing results
4. **Workflow Automation**: Simplifying the organization and execution of
   machine learning experiments

## Quick Example

Here's a simple example to demonstrate HydraFlow's basic usage:

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
def train(run: Run, cfg: Config) -> None:
    # Your training code here
    print(f"Training with lr={cfg.learning_rate}, batch_size={cfg.batch_size}")

    # Log metrics
    hydraflow.log_metric("accuracy", 0.95)

if __name__ == "__main__":
    train()
```

Run this example with:

```bash
python train.py learning_rate=0.001 batch_size=64
```

Continue to the [Installation Guide](installation.md) to get started with
HydraFlow.