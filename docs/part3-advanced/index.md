# Automating Workflows with HydraFlow

As machine learning projects grow in complexity, automating experiment
workflows becomes essential for efficiency and reproducibility. This section
covers HydraFlow's advanced features for automating and orchestrating ML
experiments.

## Overview

HydraFlow provides tools for creating automated workflows that can:

- Execute multiple experiments with different configurations
- Define dependencies between experiment steps
- Manage experiment resources efficiently
- Enable large-scale parameter sweeps
- Provide monitoring and control of experiment execution

## Key Components

The main components of HydraFlow's workflow automation tools are:

1. **Executor**: An engine for running multiple related experiments with
   defined dependencies and resource constraints.

2. **Job Scheduling**: Tools for defining, scheduling, and monitoring
   experiment jobs with complex dependencies.

3. **Scaling Mechanisms**: Features for efficiently running experiments
   at scale, including parallelization and resource management.

## Basic Workflow Example

```python
from hydraflow.executor import Executor, Job
from dataclasses import dataclass

@dataclass
class PrepConfig:
    data_path: str
    split_ratio: float = 0.2

@dataclass
class TrainConfig:
    model_type: str = "transformer"
    learning_rate: float = 0.001

# Define individual jobs
data_prep = Job(
    name="data_preparation",
    target="prepare_data.py",
    config=PrepConfig(data_path="data/raw")
)

model_training = Job(
    name="model_training",
    target="train_model.py",
    config=TrainConfig(),
    depends_on=[data_prep]  # Define dependency
)

evaluation = Job(
    name="evaluation",
    target="evaluate.py",
    depends_on=[model_training]
)

# Create and run the executor
executor = Executor()
executor.add_jobs([data_prep, model_training, evaluation])
executor.run()
```

## Advanced Features

HydraFlow's workflow automation includes several advanced features:

### Parameter Sweeps

```python
# Define parameter sweep for learning rate
learning_rates = [0.1, 0.01, 0.001, 0.0001]

train_jobs = []
for lr in learning_rates:
    job = Job(
        name=f"train_lr_{lr}",
        target="train.py",
        config=TrainConfig(learning_rate=lr)
    )
    train_jobs.append(job)

executor.add_jobs(train_jobs)
```

### Conditional Execution

```python
# Only run evaluation if training meets accuracy threshold
def eval_condition(job_result):
    return job_result.metrics.get("accuracy", 0) > 0.8

evaluation = Job(
    name="evaluation",
    target="evaluate.py",
    depends_on=[model_training],
    condition=eval_condition
)
```

### Parallel Execution

```python
# Run jobs in parallel
executor = Executor(max_parallel=4)  # Run up to 4 jobs simultaneously
executor.add_jobs(jobs)
executor.run()
```

## What's Next

In the following pages, we'll explore HydraFlow's workflow automation tools
in detail:

- [Executor](executor.md): Learn about the [`Executor`][hydraflow.executor.Executor]
  class and how to use it to orchestrate complex workflows.

- [Job Scheduling](job-scheduling.md): Discover how to define, schedule, and
  monitor experiment jobs with dependencies and conditions.

- [Scaling](scaling.md): Explore techniques for scaling experiments to handle
  large-scale parameter sweeps and optimization searches.