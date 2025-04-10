# Core Concepts

This page introduces the fundamental concepts of HydraFlow that form the
foundation of the framework.

## Run

A `Run` represents a single execution of a machine learning experiment.
Each run:

- Contains metadata about the experiment (parameters, metrics, tags)
- Links to artifacts generated during execution (models, datasets, plots)
- Provides methods to track and analyze experiment results

Example usage:

```python
from hydraflow.core import Run

# Load an existing run
run = Run.load("path/to/run")

# Access run information
print(run.info.run_id)
print(run.params)  # Configuration parameters
print(run.metrics)  # Performance metrics
```

The `Run` class serves as the primary interface for interacting with
individual experiments, allowing you to analyze results and compare
different approaches.

## RunCollection

A `RunCollection` is a collection of `Run` instances that provides tools
for analyzing and comparing multiple experiments. Key features include:

- Filtering runs based on parameters or metrics
- Grouping runs by common attributes
- Aggregating metrics across runs
- Visualizing experiment results

Example usage:

```python
from hydraflow.core import Run

# Load multiple runs
runs = Run.load(["path/to/run1", "path/to/run2", "path/to/run3"])

# Filter runs by parameter value
filtered_runs = runs.filter({"model.type": "lstm"})

# Group runs by a parameter
grouped_runs = runs.group_by("dataset.name")

# Calculate average metric across runs
avg_accuracy = runs.mean("metrics.accuracy")
```

The `RunCollection` class enables comparative analysis across multiple
experiments, making it easy to identify trends and optimal configurations.

## Configuration Management

HydraFlow uses a hierarchical configuration system based on OmegaConf and
Hydra. This provides:

- Typed configuration with schema validation
- Configuration composition from multiple sources
- Dynamic configuration resolution
- Command-line overrides

Example configuration structure:

```yaml
# config.yaml
model:
  type: transformer
  hidden_size: 512

training:
  batch_size: 32
  learning_rate: 0.001

dataset:
  name: mnist
  split: [0.8, 0.1, 0.1]
```

Loading and using configuration:

```python
from hydraflow.config import load_config

# Load configuration
cfg = load_config("config.yaml")

# Access configuration values
model_type = cfg.model.type  # "transformer"
batch_size = cfg.training.batch_size  # 32

# Configuration can be passed to Run creation
run = Run.create(cfg=cfg)
```

## Experiment Tracking

HydraFlow integrates with MLflow for experiment tracking, providing:

- Automatic logging of parameters, metrics, and artifacts
- Experiment organization and versioning
- Remote storage support
- Visualizations and dashboards

Example tracking usage:

```python
from hydraflow.tracking import log_metric, log_artifact

# Log a metric
log_metric("accuracy", 0.95)

# Log an artifact
log_artifact("model.pt", "Model checkpoint")
```

## Pipelines

HydraFlow supports creating reproducible ML pipelines that:

- Define dependencies between components
- Manage data flow between pipeline stages
- Track intermediate artifacts
- Enable selective re-execution

Example pipeline:

```python
from hydraflow.pipeline import Pipeline, Stage

# Define pipeline stages
data_stage = Stage("data_preparation", prepare_data)
train_stage = Stage("model_training", train_model)
eval_stage = Stage("model_evaluation", evaluate_model)

# Create pipeline with dependencies
pipeline = Pipeline([
    data_stage,
    train_stage.depends_on(data_stage),
    eval_stage.depends_on(train_stage)
])

# Execute pipeline
results = pipeline.run(cfg)
```

## Summary

These core concepts work together to provide a comprehensive framework for
managing machine learning experiments:

- `Run` represents individual experiments
- `RunCollection` enables comparative analysis
- Configuration management ensures reproducibility
- Experiment tracking records all relevant information
- Pipelines organize complex workflows

Understanding these fundamental concepts will help you leverage the full
power of HydraFlow for your machine learning projects.