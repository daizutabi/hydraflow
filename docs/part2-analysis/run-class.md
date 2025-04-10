# Run Class

The [`Run`][hydraflow.core.run.Run] class is a fundamental component of
HydraFlow's analysis toolkit, representing a single execution of an
experiment. It provides structured access to all data associated with
a run, including configuration, metrics, and artifacts.

## Basic Usage

To work with a run, first load it using either the constructor or the
`load` class method:

```python
from hydraflow import Run
from pathlib import Path

# Using constructor with Path object
run_dir = Path("mlruns/0/run_id")
run = Run(run_dir)

# Using load method with string path
run = Run.load("mlruns/0/run_id")

# Access run information
print(f"Run ID: {run.info.run_id}")
print(f"Job name: {run.info.job_name}")
print(f"Configuration: {run.cfg}")
```

## Access Run Data

The `Run` class provides access to various aspects of the experiment:

```python
# Access configuration
learning_rate = run.get("learning_rate")  # Access by key
model_type = run.get("model.type")  # Nested access with dot notation

# Access metrics
accuracy = run.metrics.get("accuracy")
loss = run.metrics.get("loss")

# Access parameters
batch_size = run.params.get("batch_size")

# Access tags
version = run.tags.get("version")
```

## Type-Safe Configuration Access

For better IDE integration and type checking, you can specify the configuration
type as a type parameter:

```python
from dataclasses import dataclass
from hydraflow import Run

@dataclass
class ModelConfig:
    type: str
    hidden_size: int

@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    seed: int = 42

# Create a typed Run instance
run = Run[Config](run_dir)

# Type-safe access with IDE auto-completion
model_type = run.cfg.model.type
lr = run.cfg.training.learning_rate
seed = run.cfg.seed
```

## Custom Implementation Classes

The `Run` class can be extended with custom implementation classes to add
domain-specific functionality:

```python
from pathlib import Path
from hydraflow import Run

class ModelLoader:
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir

    def load_weights(self):
        """Load the model weights from the artifacts directory."""
        return torch.load(self.artifacts_dir / "weights.pt")

    def evaluate(self, test_data):
        """Evaluate the model on test data."""
        model = self.load_weights()
        return model.evaluate(test_data)

# Create a Run with implementation
run = Run[Config, ModelLoader](run_dir, ModelLoader)

# Access implementation methods
weights = run.impl.load_weights()
results = run.impl.evaluate(test_data)
```

## Configuration-Aware Implementations

Implementation classes can optionally accept the run's configuration:

```python
class AdvancedModelLoader:
    def __init__(self, artifacts_dir: Path, cfg=None):
        self.artifacts_dir = artifacts_dir
        self.cfg = cfg

    def load_model(self):
        """Load model using configuration parameters."""
        model_type = self.cfg.model.type
        model_path = self.artifacts_dir / f"{model_type}_model.pt"
        return torch.load(model_path)

# The implementation will receive both artifacts_dir and cfg
run = Run[Config, AdvancedModelLoader](run_dir, AdvancedModelLoader)
model = run.impl.load_model()  # Uses configuration information
```

## Loading Multiple Runs

The `load` class method can load both individual runs and collections of runs:

```python
# Load a single run
run = Run.load("mlruns/0/run_id")

# Load multiple runs to create a RunCollection
runs = Run.load(["mlruns/0/run_id1", "mlruns/0/run_id2"])

# Load all runs from a directory pattern
runs = Run.load("mlruns/0/*")

# Load runs with parallel processing
runs = Run.load(run_dirs, n_jobs=4)  # Use 4 parallel jobs for loading
runs = Run.load(run_dirs, n_jobs=-1)  # Use all available CPU cores
```

## Accessing Run Information

The `Run` class provides extensive information about the experiment:

```python
# Basic run information
print(f"Run ID: {run.info.run_id}")
print(f"Experiment name: {run.info.job_name}")
print(f"Run directory: {run.info.run_dir}")

# Access MLflow run data
print(f"Start time: {run.info.start_time}")
print(f"End time: {run.info.end_time}")
print(f"Status: {run.info.status}")

# Access artifact paths
model_path = run.info.run_dir / "artifacts" / "model.pkl"
```

## Best Practices

1. **Use Type Parameters**: Specify configuration types with `Run[Config]`
   for better IDE support and type checking.

2. **Leverage Custom Implementations**: Create domain-specific implementation
   classes to encapsulate analysis logic.

3. **Use Parallel Loading**: For large numbers of runs, use the
   `n_jobs` parameter with `load` to speed up loading.

4. **Access Dot Notation**: Use the `get` method with dot notation
   (e.g., `run.get("model.type")`) to access nested configuration values.

## Summary

The [`Run`][hydraflow.core.run.Run] class provides a powerful interface for
working with experiment runs in HydraFlow. Its type-safe configuration access,
custom implementation support, and convenient loading mechanisms make it easy
to analyze and compare experiment results effectively.