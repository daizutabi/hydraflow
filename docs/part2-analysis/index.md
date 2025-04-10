# Analyzing Results with HydraFlow

After running experiments, the next critical step is analyzing and
comparing results to derive insights. This section covers HydraFlow's
powerful tools for accessing, analyzing, and visualizing experiment data.

## Overview

HydraFlow provides a comprehensive API for working with experiment results,
enabling you to:

- Load and access experiment data from MLflow runs
- Filter and group experiments based on configuration parameters
- Compare metrics across different experiment runs
- Transform experiment data into structured formats for analysis
- Create visualizations to identify patterns and trends

## Key Components

The main components of HydraFlow's analysis tools are:

1. **[`Run`][hydraflow.core.run.Run] Class**: Represents a single experiment
   run, providing access to configuration, metrics, and artifacts.

2. **[`RunCollection`][hydraflow.core.run_collection.RunCollection] Class**:
   A collection of `Run` instances with tools for filtering, grouping, and
   aggregating results.

3. **Data Analysis Integration**: Tools to convert experiment data into
   Polars DataFrames for advanced analysis.

## Basic Analysis Workflow

```python
from hydraflow import Run

# Load experiment runs
runs = Run.load(["path/to/run1", "path/to/run2", "path/to/run3"])

# Filter runs based on configuration
filtered_runs = runs.filter(learning_rate=0.01, model_type="transformer")

# Group runs by a parameter
grouped_runs = runs.group_by("batch_size")

# Convert to DataFrame for analysis
df = runs.to_frame("learning_rate", "batch_size", accuracy=lambda run: run.metrics["accuracy"])

# Perform analysis on the DataFrame
best_run = df.sort("accuracy", descending=True).first()
```

## Type-Safe Analysis

HydraFlow supports type-safe analysis through type parameters:

```python
from dataclasses import dataclass
from hydraflow import Run

@dataclass
class MyConfig:
    learning_rate: float
    batch_size: int
    model_type: str

# Load runs with type information
runs = Run[MyConfig].load("path/to/runs")

# Type-checked access to configuration
for run in runs:
    # IDE will provide auto-completion for cfg properties
    print(f"LR: {run.cfg.learning_rate}, Model: {run.cfg.model_type}")
```

## Implementation Support

HydraFlow can integrate with custom implementation classes to provide domain-specific
functionality:

```python
from hydraflow import Run
from pathlib import Path

class ModelAnalyzer:
    def __init__(self, artifacts_dir: Path, cfg=None):
        self.artifacts_dir = artifacts_dir
        self.cfg = cfg

    def load_model(self):
        return torch.load(self.artifacts_dir / "model.pt")

    def analyze_performance(self):
        # Custom analysis logic
        pass

# Load runs with implementation
runs = Run[MyConfig, ModelAnalyzer].load("path/to/runs", ModelAnalyzer)

# Access implementation methods
for run in runs:
    model = run.impl.load_model()
    results = run.impl.analyze_performance()
```

## What's Next

In the following pages, we'll explore HydraFlow's analysis tools in detail:

- [Run Class](run-class.md): Learn how to use the [`Run`][hydraflow.core.run.Run]
  class to access and analyze individual experiment runs.

- [Run Collection](run-collection.md): Discover the powerful features of the
  [`RunCollection`][hydraflow.core.run_collection.RunCollection] class for
  working with multiple runs.

- [Data Frames](data-frames.md): Learn how to convert experiment data into
  DataFrames for statistical analysis and visualization.

- [Visualization](visualization.md): Create insightful visualizations to
  understand experiment results and identify optimization opportunities.