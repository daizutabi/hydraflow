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

# Alternatively, use double underscore notation for nested access
model_type = run.get("model__type")  # Equivalent to "model.type"

# Access implementation attributes or run info
metric_value = run.get("accuracy")  # From impl or cfg
run_id = run.get("run_id")  # From RunInfo

# Access special object keys
cfg = run.get("cfg")    # Returns the complete configuration object
impl = run.get("impl")  # Returns the implementation object
info = run.get("info")  # Returns the run information object

# Provide a default value if the key doesn't exist
batch_size = run.get("batch_size", 32)

# Use a callable as default to dynamically generate values based on the run
# This is useful for derived parameters or conditional defaults
lr = run.get("learning_rate", default=lambda r: r.get("base_lr", 0.01) / 10)
```

The `get` method searches for values in the following order:

1. In the configuration (`cfg`)
2. In the implementation instance (`impl`)
3. In the run information (`info`)
4. In the run object itself (`self`)

This provides a unified access interface regardless of where the data is stored.

The double underscore notation (`__`) is automatically converted to dot notation (`.`) internally,
making it useful for nested parameter access, especially when using keyword arguments in methods
that don't allow dots in parameter names.

When providing a default value, you can use either a static value or a callable function.
If you provide a callable, it will receive the Run instance as an argument, allowing you to
create context-dependent default values that can access other run parameters or properties.
This is particularly useful for:

- Creating derived parameters that don't exist in the original configuration
- Handling schema evolution across different experiment iterations
- Providing fallbacks that depend on other configuration values
- Implementing conditional logic for parameter defaults

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

## Converting to DataFrame

To convert a Run instance to a Polars DataFrame, use the
[`to_frame`][hydraflow.core.run.Run.to_frame] method.
This method adds the Run's information as columns to the DataFrame.

```python
# Basic usage
df = run.to_frame(
    lambda r: DataFrame({"value": [1, 2, 3]}),
    "run_id",
    "experiment_name"
)

# With default values
df = run.to_frame(
    lambda r: DataFrame({"value": [1, 2, 3]}),
    "run_id",
    ("status", lambda r: "completed")
)
```

The `to_frame` method accepts the following parameters:

- `function`: A function that takes a Run instance and returns a DataFrame
- `*keys`: Keys for the Run's information to add. Accepts the following formats:
    - String: A simple key (e.g., "run_id")
    - Tuple: A tuple of (key, default value or function returning default value)

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

### Finding Runs with `iter_run_dirs`

HydraFlow provides the [`iter_run_dirs`][hydraflow.core.io.iter_run_dirs]
function to easily discover runs in your MLflow tracking directory:

```python
from hydraflow.core.io import iter_run_dirs
from hydraflow import Run

# Find all runs in the tracking directory
tracking_dir = "mlruns"
run_dirs = list(iter_run_dirs(tracking_dir))
runs = Run.load(run_dirs)

# Filter runs by experiment name
# - Use a single experiment name
runs = Run.load(iter_run_dirs(tracking_dir, "my_experiment"))

# - Use multiple experiment names (with pattern matching)
runs = Run.load(iter_run_dirs(tracking_dir, ["train_*", "eval_*"]))

# - Use a custom filtering function
def filter_experiments(name: str) -> bool:
    return name.startswith("train_") and "v2" in name

runs = Run.load(iter_run_dirs(tracking_dir, filter_experiments))
```

The `iter_run_dirs` function yields paths to run directories that can be
directly passed to `Run.load`. This makes it easy to find and load runs
based on experiment names or custom filtering criteria.

## Best Practices

1. **Use Type Parameters**: Specify configuration types with `Run[Config]`
   for better IDE support and type checking.

2. **Leverage Custom Implementations**: Create domain-specific implementation
   classes to encapsulate analysis logic.

3. **Use Parallel Loading**: For large numbers of runs, use the
   `n_jobs` parameter with `load` to speed up loading.

4. **Unified Data Access**: Use the `get` method as a unified interface
   to access data from all components (configuration, implementation, and run info).
   It provides a consistent way to retrieve values regardless of where they are stored,
   with a clear precedence order (cfg → impl → info).

5. **Default Values**: When accessing potentially missing keys, use the
   `get` method's default parameter: `run.get("key", default_value)`.

## Summary

The [`Run`][hydraflow.core.run.Run] class provides a powerful interface for
working with experiment runs in HydraFlow. Its type-safe configuration access,
custom implementation support, and convenient loading mechanisms make it easy
to analyze and compare experiment results effectively.
