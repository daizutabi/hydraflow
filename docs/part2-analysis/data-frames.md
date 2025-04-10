# Working with DataFrames

Converting experiment data to DataFrames is a powerful approach for analyzing
and visualizing results. HydraFlow integrates with [Polars](https://pola.rs/),
a fast DataFrame library, to provide flexible data manipulation capabilities.

## Converting Runs to DataFrames

The [`to_frame`][hydraflow.core.run_collection.RunCollection.to_frame] method
transforms a [`RunCollection`][hydraflow.core.run_collection.RunCollection]
into a Polars DataFrame:

```python
from hydraflow import Run

# Load multiple runs
runs = Run.load("mlruns/0/*")

# Basic conversion with default columns
df = runs.to_frame()
print(df.columns)  # Includes run_id, job_name, status, etc.
```

## Selecting Configuration Parameters

You can specify which configuration parameters to include as columns:

```python
# Include specific parameters
df = runs.to_frame("learning_rate", "batch_size", "model.type")

# Include all parameters (can be inefficient for large configurations)
df = runs.to_frame(*runs.first().cfg.keys())
```

## Including Metrics and Computed Values

Add metrics and computed values using callable arguments:

```python
# Add metrics as columns
df = runs.to_frame(
    "learning_rate",
    "batch_size",
    accuracy=lambda run: run.metrics.get("accuracy", 0),
    loss=lambda run: run.metrics.get("loss", float("inf"))
)

# Add computed values
df = runs.to_frame(
    "learning_rate",
    lr_reciprocal=lambda run: 1/run.get("learning_rate"),
    training_time=lambda run: (run.info.end_time - run.info.start_time).total_seconds()
)
```

## Handling Complex Return Types

The callables in `to_frame` can return various types of data:

### Returning Lists

When a callable returns a list, each element becomes a separate column:

```python
def metrics_list(run):
    return [
        run.metrics.get("accuracy", 0),
        run.metrics.get("precision", 0),
        run.metrics.get("recall", 0)
    ]

df = runs.to_frame(
    "learning_rate",
    metrics=metrics_list  # Creates columns: metrics_0, metrics_1, metrics_2
)
```

### Returning Dictionaries

When a callable returns a dictionary, each key-value pair becomes a column:

```python
def get_metrics(run):
    return {
        "accuracy": run.metrics.get("accuracy", 0),
        "precision": run.metrics.get("precision", 0),
        "recall": run.metrics.get("recall", 0),
        "f1": run.metrics.get("f1", 0)
    }

df = runs.to_frame(
    "learning_rate",
    "batch_size",
    metrics=get_metrics  # Creates columns: metrics_accuracy, metrics_precision, etc.
)
```

### Custom Column Names

You can customize prefix behavior for dictionary and list returns:

```python
# No prefix for dictionary keys
df = runs.to_frame(
    "learning_rate",
    **get_metrics()  # Creates columns: accuracy, precision, recall, f1
)

# Custom prefix for list elements
df = runs.to_frame(
    "learning_rate",
    evaluation_metrics=metrics_list  # Creates: evaluation_metrics_0, evaluation_metrics_1, etc.
)
```

## Working with Nested Configurations

For nested configurations, use dot notation or custom functions:

```python
# Using dot notation
df = runs.to_frame(
    "model.type",
    "model.hidden_size",
    "optimizer.learning_rate"
)

# Using custom functions for better control
df = runs.to_frame(
    model_type=lambda run: run.cfg.model.type,
    hidden_size=lambda run: run.cfg.model.hidden_size,
    learning_rate=lambda run: run.cfg.optimizer.learning_rate
)
```

## Accessing Implementation Data

If your runs use custom implementation classes, you can access their data:

```python
class ModelAnalyzer:
    def __init__(self, artifacts_dir, cfg=None):
        self.artifacts_dir = artifacts_dir
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = torch.load(self.artifacts_dir / "model.pt")
        return self.model

    def count_parameters(self):
        model = self.load_model()
        return sum(p.numel() for p in model.parameters())

# Include implementation data in DataFrame
runs = Run[Config, ModelAnalyzer].load(run_dirs, ModelAnalyzer)
df = runs.to_frame(
    "model.type",
    "learning_rate",
    param_count=lambda run: run.impl.count_parameters(),
    model_size_mb=lambda run: os.path.getsize(run.info.run_dir / "artifacts/model.pt") / (1024*1024)
)
```

## Aggregating Data with Group By

For aggregated analysis, combine `group_by` with DataFrame conversion:

```python
# Group by model type and calculate statistics
result_df = runs.group_by(
    "model.type",
    count=len,
    avg_accuracy=lambda runs: runs.mean("accuracy"),
    max_accuracy=lambda runs: runs.max("accuracy"),
    min_accuracy=lambda runs: runs.min("accuracy"),
    std_accuracy=lambda runs: runs.std("accuracy")
)

# Group by multiple parameters
multi_group = runs.group_by(
    "model.type",
    "optimizer.name",
    count=len,
    avg_accuracy=lambda runs: runs.mean("accuracy")
)
```

## Data Analysis with Polars

Once you have a DataFrame, you can leverage Polars' powerful features:

```python
import polars as pl

# Basic analysis
print(df.describe())
print(df.select(pl.col("accuracy").mean()))

# Filtering
best_models = df.filter(pl.col("accuracy") > 0.9)

# Sorting
ranked_models = df.sort("accuracy", descending=True)

# Grouping and aggregation
by_model = df.group_by("model_type").agg([
    pl.col("accuracy").mean().alias("avg_accuracy"),
    pl.col("accuracy").max().alias("max_accuracy"),
    pl.count().alias("count")
])

# Joining with other data
metadata = pl.DataFrame({
    "model_type": ["transformer", "cnn", "lstm"],
    "paper_reference": ["Vaswani et al.", "LeCun et al.", "Hochreiter & Schmidhuber"]
})
df_with_refs = df.join(metadata, on="model_type")
```

## Exporting DataFrames

Export DataFrames to various formats for sharing or further analysis:

```python
# Save to CSV
df.write_csv("experiment_results.csv")

# Save to Parquet (efficient binary format)
df.write_parquet("experiment_results.parquet")

# Save to JSON
df.write_json("experiment_results.json")

# Convert to pandas DataFrame (if needed)
pandas_df = df.to_pandas()
```

## Combining with Visualization Libraries

Use the DataFrame with visualization libraries:

```python
import plotly.express as px
import matplotlib.pyplot as plt

# Using Plotly with Polars
fig = px.scatter(
    df.to_pandas(),  # Convert to pandas for plotly
    x="learning_rate",
    y="accuracy",
    color="model_type",
    size="batch_size",
    hover_data=["run_id"]
)
fig.show()

# Using matplotlib with Polars
plt.figure(figsize=(10, 6))
for model_type, group in df.group_by("model_type"):
    plt.scatter(
        group["learning_rate"],
        group["accuracy"],
        label=model_type
    )
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_vs_lr.png")
```

## Best Practices

1. **Select Only Needed Columns**: Include only the columns you need to
   avoid large and unwieldy DataFrames.

2. **Use Callables for Complex Data**: Leverage callable arguments to
   transform and extract the exact data you need.

3. **Handle Missing Values**: Always provide defaults when accessing metrics
   or parameters that might be missing.

4. **Group First, Then Analyze**: For large datasets, use `group_by` before
   converting to a DataFrame to reduce data size.

5. **Use Polars Features**: Take advantage of Polars' optimized operations
   for faster analysis of large datasets.

## Summary

Converting experiment data to DataFrames is a powerful approach for analyzing
HydraFlow runs. The integration with Polars provides efficient and flexible
tools for extracting insights from your machine learning experiments, enabling
everything from simple parameter comparisons to complex statistical analyses.