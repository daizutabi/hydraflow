# Run Collection

The [`RunCollection`][hydraflow.core.run_collection.RunCollection] class is a
powerful tool for working with multiple experiment runs. It provides methods
for filtering, grouping, and analyzing sets of [`Run`][hydraflow.core.run.Run]
instances, making it easy to compare and extract insights from your experiments.

## Architecture

`RunCollection` is built on top of the more general
[`Collection`][hydraflow.core.collection.Collection]
class, which provides a flexible foundation for working with sequences
of items. This architecture offers several benefits:

1. **Consistent Interface**: All collection-based classes in HydraFlow
    share a common interface and behavior
2. **Code Reuse**: Core functionality is implemented once in the base
    class and inherited by specialized collections
3. **Extensibility**: New collection types can easily be created
    for different item types
4. **Type Safety**: Generic type parameters ensure type checking
    throughout the collection hierarchy

The `Collection` class implements the Python `Sequence` protocol,
allowing it to be used like standard Python collections (lists, tuples)
while providing specialized methods for filtering, grouping, and data extraction.

`RunCollection` extends this foundation with run-specific functionality,
particularly for working with MLflow experiment data. This layered
design separates generic collection behavior from domain-specific operations.

## Creating a Run Collection

There are several ways to create a `RunCollection`:

```python
from hydraflow import Run, RunCollection
from pathlib import Path

# Method 1: Using Run.load with multiple paths
run_dirs = ["mlruns/exp_id/run_id1", "mlruns/exp_id/run_id2"]
runs = Run.load(run_dirs)

# Method 2: Using a generator expression
run_dirs = Path("mlruns/exp_id").glob("*")
runs = Run.load(run_dirs)

# Method 3: Creating from a list of Run instances
run1 = Run(Path("mlruns/exp_id/run_id1"))
run2 = Run(Path("mlruns/exp_id/run_id2"))
runs = RunCollection([run1, run2])

# Method 4: Using iter_run_dirs to find runs dynamically
from hydraflow import iter_run_dirs

# Find all runs in a tracking directory
tracking_dir = "mlruns"
runs = Run.load(iter_run_dirs(tracking_dir))

# Find runs from specific experiments
runs = Run.load(iter_run_dirs(tracking_dir, ["experiment1", "experiment2"]))

# Use pattern matching for experiment names
runs = Run.load(iter_run_dirs(tracking_dir, "transformer_*"))

# Use a custom filter function for experiment names
def is_recent_version(name: str) -> bool:
    return name.startswith("model_") and "v2" in name

runs = Run.load(iter_run_dirs(tracking_dir, is_recent_version))
```

## Basic Operations

The `RunCollection` class supports common operations for working with collections:

```python
# Check the number of runs
print(f"Number of runs: {len(runs)}")

# Iterate over runs
for run in runs:
    print(f"Run ID: {run.info.run_id}")

# Access individual runs by index
first_run = runs[0]
last_run = runs[-1]

# Slice the collection
subset = runs[1:4]  # Get runs 1, 2, and 3
```

## Filtering Runs

One of the most powerful features of `RunCollection` is the ability to filter
runs based on configuration parameters or other criteria:

```python
# Filter by exact parameter value
transformer_runs = runs.filter(model_type="transformer")

# Filter with multiple conditions (AND logic)
specific_runs = runs.filter(
    model_type="transformer",
    learning_rate=0.001,
    batch_size=32
)

# Filter with dot notation for nested parameters
# Use a tuple to specify the parameter name and value
nested_filter = runs.filter(("model.hidden_size", 512))

# Filter with double underscore notation for nested parameters
# This is often more convenient with keyword arguments
nested_filter = runs.filter(model__hidden_size=512)  # Equivalent to "model.hidden_size"
nested_filter = runs.filter(model__encoder__num_layers=6)  # For deeply nested parameters

# Filter with tuple for range values (inclusive)
lr_range = runs.filter(learning_rate=(0.0001, 0.01))

# Filter with list for multiple allowed values (OR logic)
multiple_models = runs.filter(model_type=["transformer", "lstm"])

# Filter by a predicate function
def is_large_image(run: Run):
    return run.get("width") + run.get("height") > 100

good_runs = runs.filter(is_large_image)
```

The double underscore notation (`__`) is particularly useful for accessing nested
configuration parameters with keyword arguments, as it's automatically converted to
dot notation (`.`) internally. This allows you to write more natural and Pythonic
filtering expressions, especially for deeply nested configurations.

## Advanced Filtering

The `filter` method supports more complex filtering patterns:

```python
# Combine different filter types
complex_filter = runs.filter(
    model_type=["transformer", "lstm"],
    learning_rate=(0.0001, 0.01),
    batch_size=32
)

# Chained filtering
final_runs = runs.filter(model_type="transformer").filter(learning_rate=0.001)

# Advanced filtering using predicate functions with callable defaults
# This example filters runs based on learning rate efficiency (lr * batch_size)
# Even if some runs are missing one parameter, the default logic provides values
def has_efficient_lr(run: Run) -> bool:
    lr = run.get("learning_rate", default=lambda r: r.get("base_lr", 0.01) * r.get("lr_multiplier", 1.0))
    batch_size = run.get("batch_size", default=lambda r: r.get("default_batch_size", 32))
    return lr * batch_size < 0.5

# Apply the complex predicate
efficient_runs = runs.filter(has_efficient_lr)
```

The combination of predicate functions with callable defaults in `get` enables sophisticated
filtering logic that can handle missing parameters and varied configuration schemas across
different experiment runs.

## Sorting Runs

The `sort` method allows you to sort runs based on specific criteria:

```python
# Sort by accuracy in descending order
runs.sort("learning_rate", reverse=True)

# Sort by multiple keys
runs.sort("learning_rate", "model_type")
```

## Getting Individual Runs

While `filter` returns a `RunCollection`, the `get` method returns a single
`Run` instance that matches the criteria:

```python
# Get a specific run (raises error if multiple or no matches are found)
best_run = runs.get(model_type="transformer", learning_rate=0.001)

# Try to get a specific run. If no match is found, return None
fallback_run = runs.try_get(model_type="transformer")

# Get the first matching run.
first_match = runs.first(model_type="transformer")

# Get the last matching run.
last_match = runs.last(model_type="transformer")
```

## Extracting Data

RunCollection provides several methods to extract specific data from runs:

```python
# Extract values for a specific key as a list
learning_rates = runs.to_list("learning_rate")

# Extract values with a static default for missing values
batch_sizes = runs.to_list("batch_size", default=32)

# Extract values with a callable default that dynamically computes values
# This is particularly useful for handling missing parameters or derived values
accuracies = runs.to_list("accuracy", default=lambda run: run.get("val_accuracy", 0.0) * 0.9)

# Extract values as a NumPy array
batch_sizes = runs.to_numpy("batch_size")

# Extract with callable default for complex scenarios
learning_rates = runs.to_numpy(
    "learning_rate",
    default=lambda run: run.get("base_lr", 0.01) * run.get("lr_schedule_factor", 1.0)
)

# Extract values as a Polars Series
lr_series = runs.to_series("learning_rate")

# Extract with a custom name for the series
model_series = runs.to_series("model_type", name="Model Architecture")

# Extract with callable default and custom name
effective_lr = runs.to_series(
    "learning_rate",
    default=lambda run: run.get("base_lr", 0.01) * run.get("lr_multiplier", 1.0),
    name="Effective Learning Rate"
)

# Use Series for further analysis and operations
import polars as pl
# Combine multiple series into a DataFrame
df = pl.DataFrame([
    runs.to_series("model_type", name="Model"),
    runs.to_series("batch_size", default=32, name="Batch Size"),
    effective_lr
])
# Perform operations between Series
normalized_acc = runs.to_series("accuracy", default=0.0, name="Accuracy")
efficiency = normalized_acc / effective_lr  # Series division

# Get unique values for a key
model_types = runs.unique("model_type")

# Count unique values
num_model_types = runs.n_unique("model_type")
```

All data extraction methods (`to_list`, `to_numpy`, `to_series`, etc.)
support both static and callable default values,
matching the behavior of the `Run.get` method. When using a callable default,
the function receives the Run instance as an argument, allowing you to:

- Implement fallback logic for missing parameters
- Create derived values based on multiple parameters
- Handle varying configuration schemas across different experiments
- Apply transformations to the raw parameter values

This makes it much easier to work with heterogeneous collections of
runs that might have different parameter sets or evolving configuration
schemas.

## Converting to DataFrame

For advanced analysis, you can convert your runs to a Polars DataFrame:

```python
# DataFrame with run information and entire configuration
df = runs.to_frame()

# DataFrame with specific configuration parameters
df = runs.to_frame("model_type", "learning_rate", "batch_size")

# Include Run, configuration, or implementation objects as columns
df = runs.to_frame("model_type", "learning_rate", "run")  # Include Run objects
df = runs.to_frame("model_type", "cfg")  # Include configuration objects
df = runs.to_frame("run_id", "run", "cfg", "impl")  # Include all objects

# Specify default values for missing parameters using the defaults parameter
df = runs.to_frame(
    "model_type",
    "learning_rate",
    "batch_size",
    defaults={"learning_rate": 0.01, "batch_size": 32}
)

# Missing values without defaults are represented as None (null) in the DataFrame
# This allows for standard handling of missing data in Polars
missing_values_df = runs.to_frame("model_type", "parameter_that_might_be_missing")

# Filter rows with non-null values
import polars as pl
valid_rows = missing_values_df.filter(pl.col("parameter_that_might_be_missing").is_not_null())

# Fill null values after creating the DataFrame
filled_df = missing_values_df.with_columns(
    pl.col("parameter_that_might_be_missing").fill_null("default_value")
)
```

## Concatenating Multiple Runs

To convert and concatenate multiple Run instances into a DataFrame,
use the [`concat`][hydraflow.core.run_collection.RunCollection.concat] method.
This method adds each Run's information as columns to the DataFrame and concatenates them.

```python
# Basic usage
df = run_collection.concat(
    lambda r: DataFrame({"value": [1, 2, 3]}),
    "run_id",
    "experiment_name"
)

# With default values
df = run_collection.concat(
    lambda r: DataFrame({"value": [1, 2, 3]}),
    "run_id",
    ("status", lambda r: "completed")
)
```

The `concat` method accepts the following parameters:

- `function`: A function that takes each Run instance and returns a DataFrame
- `*keys`: Keys for the Run's information to add. Accepts the following formats:
    - String: A simple key (e.g., "run_id")
    - Tuple: A tuple of (key, default value or function returning default value)

## Grouping Runs

The `group_by` method allows you to organize runs based on parameter values:

```python
# Group by a single parameter
model_groups = runs.group_by("model_type")

# Group by nested parameter using dot notation
architecture_groups = runs.group_by("model.architecture")

# Iterate through groups
for model_type, group in model_groups.items():
    print(f"Model type: {model_type}, Runs: {len(group)}")

# Group by multiple parameters
param_groups = runs.group_by("model_type", "learning_rate")

# Mix of regular and nested parameters using double underscore notation
param_groups = runs.group_by("model_type", "model__hidden_size", "optimizer__learning_rate")

# Access a specific group
transformer_001_group = param_groups[("transformer", 0.001)]

# Aggregating grouped runs using the agg method
# This returns a DataFrame with the aggregated results
model_counts = model_groups.agg(count=lambda runs: len(runs))
model_avg_loss = model_groups.agg(
    avg_loss=lambda runs: sum(run.get("loss", 0) for run in runs) / len(runs),
    min_loss=lambda runs: min(run.get("loss", float("inf")) for run in runs)
)
```

The `group_by` method returns a `GroupBy` instance that maps keys to
`RunCollection` instances. This design allows you to:

- Work with each group as a separate `RunCollection` with all the
  filtering, sorting, and analysis capabilities
- Perform custom operations on each group that might not be expressible
  as simple aggregation functions
- Chain additional operations on specific groups that interest you
- Implement multi-stage analysis workflows where you need to maintain
  the full run information at each step

To perform aggregations on the grouped data, use the `agg` method on
the GroupBy instance. This transforms the grouped data into a DataFrame
with aggregated results.
You can define multiple aggregation functions to compute different
metrics across each group.

This approach preserves all information in each group, giving
you maximum flexibility for downstream analysis.

## Type-Safe Run Collections

Like the `Run` class, `RunCollection` supports type parameters for better
IDE integration:

```python
from dataclasses import dataclass
from hydraflow import Run, RunCollection

@dataclass
class ModelConfig:
    type: str
    hidden_size: int

@dataclass
class Config:
    model: ModelConfig
    learning_rate: float
    batch_size: int

# Create a typed RunCollection
run_dirs = ["mlruns/exp_id/run_id1", "mlruns/exp_id/run_id2"]
runs = Run[Config].load(run_dirs)

# Type-safe access in iterations
for run in runs:
    # IDE will provide auto-completion
    model_type = run.cfg.model.type
    lr = run.cfg.learning_rate
```

## Implementation-Aware Collections

You can also create collections with custom implementation classes:

```python
class ModelAnalyzer:
    def __init__(self, artifacts_dir: Path, cfg: Config | None = None):
        self.artifacts_dir = artifacts_dir
        self.cfg = cfg

    def load_model(self):
        # Load the model from artifacts
        pass

    def evaluate(self, data):
        # Evaluate the model
        pass

# Create a collection with implementation
runs = Run[Config, ModelAnalyzer].load(run_dirs, ModelAnalyzer)

# Access implementation methods
for run in runs:
    model = run.impl.load_model()
    results = run.impl.evaluate(test_data)
```

## Best Practices

1. **Filter Early**: Apply filters as early as possible
   to reduce the number of runs you're working with.

2. **Use Type Parameters**: Specify
   configuration/implementation types
   with `Run[Config]` or `Run[Config, Impl]` and
   use `load` method to collect runs for better IDE support and
   type checking.

3. **Chain Operations**: Combine filtering, grouping,
   and object extraction for efficient analysis workflows.

4. **Use DataFrame Integration**: Convert to DataFrames
   for complex analysis and visualization needs.

## Summary

The [`RunCollection`][hydraflow.core.run_collection.RunCollection] class is a
powerful tool for comparative analysis of machine learning experiments. Its
filtering, grouping, and data extraction capabilities enable efficient extraction
of insights from large sets of experiments, helping you identify optimal
configurations and understand performance trends.

[hydraflow.core.collection.Collection]: ../../api/hydraflow/core/collection.html#hydraflow.core.collection.Collection
