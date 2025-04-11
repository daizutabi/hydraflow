# Run Class

The [`Run`][hydraflow.core.run.Run] class is a fundamental component of
HydraFlow's analysis toolkit, representing a single execution of an
experiment. It provides structured access to all data associated with
a run, including configuration and artifacts.

## Basic Usage

To work with a run, first load it using either
the constructor or the
[`Run.load`][hydraflow.core.run.Run.load] class method:

```python
from hydraflow import Run
from pathlib import Path

# Using constructor with Path object
run_dir = Path("mlruns/exp_id/run_id")
run = Run(run_dir)

# Using load method with string path
run = Run.load("mlruns/exp_id/run_id")
```

## Access Run Data

The `Run` class provides access to run information and configuration.

### Run Information

The `info` attribute provides the following information:

```python
print(f"Run ID: {run.info.run_id}")
print(f"Run Directory: {run.info.run_dir}")
print(f"Job name: {run.info.job_name}")
```

### Run Configuration

The `cfg` attribute provides the entire configuration:

```python
# Access entire configuration
print(f"Configuration: {run.cfg}")
```

You can also access configuration values by key using
the [`get`][hydraflow.core.run.Run.get] method:

```python
# Access configuration by key
learning_rate = run.get("learning_rate")

# Nested access with dot notation
model_type = run.get("model.type")
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

The `Run` class can be extended with custom
implementation classes to add
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
```

The `impl` attribute provides access to the
implementation class instance:

```python
# Access implementation methods
weights = run.impl.load_weights()
results = run.impl.evaluate(test_data)
```

## Configuration-Aware Implementations

Implementation classes can optionally accept the run's configuration:

```python
class AdvancedModelLoader:
    def __init__(self, artifacts_dir: Path, cfg: Config | None = None):
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
run = Run.load("mlruns/exp_id/run_id")

# Load multiple runs to create a RunCollection
run_dirs = ["mlruns/exp_id/run_id1", "mlruns/exp_id/run_id2"]
runs = Run.load(run_dirs)

# Load runs with parallel processing
runs = Run.load(run_dirs, n_jobs=4)  # Use 4 parallel jobs for loading
runs = Run.load(run_dirs, n_jobs=-1)  # Use all available CPU cores
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