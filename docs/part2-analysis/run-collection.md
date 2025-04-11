# Run Collection

The [`RunCollection`][hydraflow.core.run_collection.RunCollection] class is a
powerful tool for working with multiple experiment runs. It provides methods
for filtering, grouping, and analyzing sets of [`Run`][hydraflow.core.run.Run]
instances, making it easy to compare and extract insights from your experiments.

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

# Filter with tuple for range values (inclusive)
lr_range = runs.filter(learning_rate=(0.0001, 0.01))

# Filter with list for multiple allowed values (OR logic)
multiple_models = runs.filter(model_type=["transformer", "lstm"])

# Filter by a predicate function
def is_large_image(run: Run):
    return run.get("width") + run.get("height") > 100

good_runs = runs.filter(predicate=is_large_image)
```

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
```

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

# Extract values as a NumPy array
batch_sizes = runs.to_numpy("batch_size")

# Get unique values for a key
model_types = runs.unique("model_type")

# Count unique values
num_model_types = runs.n_unique("model_type")
```

## Converting to DataFrame

For advanced analysis, you can convert your runs to a Polars DataFrame:

```python
# DataFrame with run information and entire configuration
df = runs.to_frame()

# DataFrame with specific configuration parameters
df = runs.to_frame("model_type", "learning_rate", "batch_size")

# Using a custom function that returns multiple columns
def get_metrics(run: Run) -> dict[str, float]:
    return {
        "accuracy": run.impl.accuracy(),
        "precision": run.impl.precision(),
    }

df = runs.to_frame("model_type", metrics=get_metrics)
```

## Grouping Runs

The `group_by` method allows you to organize runs based on parameter values:

```python
# Group by a single parameter
model_groups = runs.group_by("model_type")

# Iterate through groups
for model_type, group in model_groups.items():
    print(f"Model type: {model_type}, Runs: {len(group)}")

# Group by multiple parameters
param_groups = runs.group_by("model_type", "learning_rate")

# Access a specific group
transformer_001_group = param_groups[("transformer", 0.001)]
```

When no aggregation functions are provided, `group_by` returns a dictionary mapping keys to `RunCollection` instances. This intentional design allows you to:

- Work with each group as a separate `RunCollection` with all the filtering, sorting, and analysis capabilities
- Perform custom operations on each group that might not be expressible as simple aggregation functions
- Chain additional operations on specific groups that interest you
- Implement multi-stage analysis workflows where you need to maintain the full run information at each step

This approach preserves all information in each group, giving you maximum flexibility for downstream analysis.

## Aggregation with Group By

Combine `group_by` with aggregation for powerful analysis:

```python
# Group by model type and calculate average accuracy
model_accuracies = runs.group_by(
    "model_type",
    accuracy=mean_accuracy
)

# Group by multiple parameters with multiple aggregations
results = runs.group_by(
    "model_type",
    "learning_rate",
    count=len,
    accuracy=mean_accuracy
)
```

When aggregation functions are provided as keyword arguments, `group_by` returns a Polars DataFrame with the group keys and aggregated values. This design choice offers several advantages:

- Directly produces analysis-ready results with all aggregations computed in a single operation
- Enables efficient downstream analysis using Polars' powerful DataFrame operations
- Simplifies visualization and reporting workflows
- Reduces memory usage by computing only the requested aggregations rather than maintaining full RunCollections
- Creates a clean interface that separates grouping from additional analysis steps

The DataFrame output is particularly useful for final analysis steps where you need to summarize results across many runs or prepare data for visualization.

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
   and aggregation for efficient analysis workflows.

4. **Use DataFrame Integration**: Convert to DataFrames
   for complex analysis and visualization needs.

## Summary

The [`RunCollection`][hydraflow.core.run_collection.RunCollection] class is a
powerful tool for comparative analysis of machine learning experiments. Its
filtering, grouping, and aggregation capabilities enable efficient extraction
of insights from large sets of experiments, helping you identify optimal
configurations and understand performance trends.