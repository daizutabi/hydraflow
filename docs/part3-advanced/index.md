# Advanced Multi-Run Workflows

HydraFlow extends Hydra's capabilities with advanced features for efficient
multi-run workflows. While Hydra provides basic parameter sweeping, HydraFlow
offers tools for more complex scenarios, including extended sweep syntax,
job definitions, and cluster submission.

## Overview

Part 3 focuses on HydraFlow's advanced features for efficient multi-run execution:

- **Extended Sweep Syntax** - Define complex parameter spaces with numerical
  ranges, combinations, and engineering notation

- **Job Configuration** - Create reusable job definitions with GitHub
  Actions-like syntax

- **Batch Submission** - Submit multiple runs efficiently to clusters
  and job schedulers

These capabilities help you manage complex machine learning experiments
at scale, making it easier to explore large parameter spaces and run
on distributed computing resources.

## Key Components

HydraFlow's advanced workflow capabilities are built around these components:

1. **Parser** - A powerful syntax parser for complex parameter sweeps,
   supporting ranges, lists, and combinations

2. **Job Definition** - YAML-based configuration for defining reusable
   multi-step jobs

3. **CLI Commands** - Commands for managing and executing jobs defined
   in configuration files

## Basic Workflow

HydraFlow's workflow management centers around the `hydraflow` CLI, which
provides a structured approach to executing complex experiment jobs.

### 1. Define a Job

Create a `hydraflow.yaml` file in your project directory to define reusable job
configurations:

```yaml title="hydraflow.yaml"
jobs:
  train:
    run: python train.py
    steps:
      # Each combination creates a separate execution
      - batch: >-
          model=small,large
          learning_rate=0.1,0.2

      # Parameters included in each batch execution
      - args: optimizer=adam,sgd
```

This configuration defines a job named `train` that will execute `train.py` with
different parameter combinations. The `batch` step creates a grid of parameters
(model and learning rate), while the `args` step adds parameters to each execution.

### 2. Validate with Dry Run

Before executing, validate your job configuration with the `--dry-run` flag:

```bash
$ hydraflow run train --dry-run
```

This command displays the exact commands that would be executed without actually
running them, allowing you to verify parameter combinations and execution flow.

### 3. Execute the Job

Once validated, run the job with:

```bash
$ hydraflow run train
```

This executes the following commands sequentially:

```bash
$ python train.py model=small learning_rate=0.1 optimizer=adam,sgd
$ python train.py model=large learning_rate=0.1 optimizer=adam,sgd
$ python train.py model=small learning_rate=0.2 optimizer=adam,sgd
$ python train.py model=large learning_rate=0.2 optimizer=adam,sgd
```

When `train.py` is a Hydra application, it will perform an inner sweep on the `optimizer`
parameter while keeping `model` and `learning_rate` fixed for each run. This creates
a total of 8 combinations (2 models × 2 learning rates × 2 optimizers).

### Parallelize with Submission Commands

The true power of HydraFlow's workflow management emerges when using job submission
commands like `sbatch` or `qsub`. Instead of running sequentially, you can submit
jobs to run in parallel on a cluster:

```yaml title="hydraflow.yaml"
jobs:
  train:
    submit: sbatch --partition=gpu --nodes=1 job.sh
    steps:
      - batch: >-
          model=small,large
          learning_rate=0.1,0.2
      - args: optimizer=adam,sgd
```

This approach offers several advantages:

- **Parallelization**: Execute multiple parameter combinations simultaneously
- **Resource Optimization**: Allocate appropriate resources to each job
- **Scalability**: Easily scale to hundreds or thousands of experiments
- **Fault Isolation**: Failures in one job don't affect others

This separation of batch parameters and inner-sweep parameters allows for efficient
resource allocation while maintaining the flexibility of Hydra's parameter sweeping.

## When to Use Advanced Workflows

Consider using HydraFlow's advanced workflow features when:

- You need to explore complex parameter spaces with many configurations
- Your experiments require a structured approach with multiple steps
- You want to leverage distributed computing resources
- You need reproducible workflow definitions

## What's Next

The following pages explain HydraFlow's advanced features in detail:

- [Extended Sweep Syntax](sweep-syntax.md) - Learn how to define complex
  parameter spaces using HydraFlow's extended syntax

- [Job Configuration](job-configuration.md) - Define reusable and maintainable
  job configurations with steps

- [Batch Submission](batch-submission.md) - Run jobs efficiently on
  clusters and job schedulers