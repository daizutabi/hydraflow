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
runs = Run.load(["mlruns/0/run_id1", "mlruns/0/run_id2", "mlruns/0/run_id3"])

# Method 2: Using a pattern to find multiple runs
runs = Run.load("mlruns/0/*")  # Load all runs in experiment 0

# Method 3: Creating from a list of Run instances
run1 = Run(Path("mlruns/0/run_id1"))
run2 = Run(Path("mlruns/0/run_id2"))
runs = RunCollection([run1, run2])

# Method 4: Using a generator expression
run_dirs = Path("mlruns/0").glob("*")
runs = Run.load(run_dirs)
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
nested_filter = runs.filter(model__hidden_size=512)  # Double underscore for nesting

# Filter with tuple for range values (inclusive)
lr_range = runs.filter(learning_rate=(0.0001, 0.01))

# Filter with list for multiple allowed values (OR logic)
multiple_models = runs.filter(model_type=["transformer", "lstm"])

# Filter by a predicate function
def high_accuracy(run):
    return run.metrics.get("accuracy", 0) > 0.9

good_runs = runs.filter(predicate=high_accuracy)
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

# Filter based on metrics
accurate_runs = runs.filter(lambda run: run.metrics.get("accuracy", 0) > 0.9)

# Chained filtering
final_runs = runs.filter(model_type="transformer").filter(learning_rate=0.001)
```

## Getting Individual Runs

While `filter` returns a `RunCollection`, the `get` method returns a single
`Run` instance that matches the criteria:

```python
# Get a specific run (raises error if multiple or no matches are found)
best_run = runs.get(model_type="transformer", learning_rate=0.001)

# Get with a default value if no match is found
fallback_run = runs.get(model_type="unknown_model", default=None)

# Get the first matching run when multiple matches exist
first_match = runs.get(model_type="transformer", first=True)
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

## Aggregating Data

`RunCollection` provides methods to aggregate data across runs:

```python
# Calculate average metric value
avg_accuracy = runs.mean("accuracy")

# Calculate other statistics
max_accuracy = runs.max("accuracy")
min_accuracy = runs.min("accuracy")
accuracy_std = runs.std("accuracy")

# Calculate with a custom aggregation function
q75_accuracy = runs.aggregate("accuracy", lambda values: np.percentile(values, 75))
```

## Converting to DataFrame

For advanced analysis, you can convert your runs to a Polars DataFrame:

```python
# Basic DataFrame with run information
df = runs.to_frame()

# Include specific configuration parameters
df = runs.to_frame("model_type", "learning_rate", "batch_size")

# Include metrics
df = runs.to_frame(
    "model_type",
    "learning_rate",
    accuracy=lambda run: run.metrics.get("accuracy", 0),
    loss=lambda run: run.metrics.get("loss", float("inf"))
)

# Using a custom function that returns multiple columns
def get_metrics(run):
    return {
        "accuracy": run.metrics.get("accuracy", 0),
        "precision": run.metrics.get("precision", 0),
        "recall": run.metrics.get("recall", 0),
        "f1": run.metrics.get("f1", 0)
    }

df = runs.to_frame("model_type", "learning_rate", metrics=get_metrics)
```

## Aggregation with Group By

Combine `group_by` with aggregation for powerful analysis:

```python
# Group by model type and calculate average accuracy
model_accuracies = runs.group_by("model_type", accuracy=lambda runs: runs.mean("accuracy"))

# Group by multiple parameters with multiple aggregations
results = runs.group_by(
    "model_type",
    "learning_rate",
    count=len,
    avg_accuracy=lambda runs: runs.mean("accuracy"),
    max_accuracy=lambda runs: runs.max("accuracy")
)
```

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
runs = Run[Config].load(["mlruns/0/run_id1", "mlruns/0/run_id2"])

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
    def __init__(self, artifacts_dir, cfg=None):
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

## Parallel Processing

For operations on large collections, you can use parallel processing:

```python
# Load runs in parallel
runs = Run.load(run_dirs, n_jobs=4)  # Use 4 processes
runs = Run.load(run_dirs, n_jobs=-1)  # Use all available cores

# Process runs in parallel with custom function
def process_run(run):
    model = load_model(run)
    return evaluate_model(model)

results = runs.map(process_run, n_jobs=4)
```

## Best Practices

1. **Filter Early**: Apply filters as early as possible to reduce the number of
   runs you're working with.

2. **Use Type Parameters**: Specify configuration types with `Run[Config]` for
   better IDE support and type checking.

3. **Chain Operations**: Combine filtering, grouping, and aggregation for
   efficient analysis workflows.

4. **Use DataFrame Integration**: Convert to DataFrames for complex analysis and
   visualization needs.

5. **Leverage Parallelism**: Use parallel processing for operations on large
   collections of runs.

## Summary

The [`RunCollection`][hydraflow.core.run_collection.RunCollection] class is a
powerful tool for comparative analysis of machine learning experiments. Its
filtering, grouping, and aggregation capabilities enable efficient extraction
of insights from large sets of experiments, helping you identify optimal
configurations and understand performance trends.