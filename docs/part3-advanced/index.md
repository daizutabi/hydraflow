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
   job configurations with independent parameter sets

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
    sets:
      # First independent parameter set
      - each: >-
          model=small,large
          learning_rate=0.1,0.2

      # Second independent parameter set
      - each: >-
          optimizer=adam,sgd
          dropout=0.1,0.2
```

This configuration defines a job named `train` that will execute `train.py` with
different parameter combinations. Each `set` represents an independent set of
parameter combinations - the sets do not build upon or depend on each other.

### 2. Understanding Sets

Sets in HydraFlow are completely independent from each other. Each set generates
its own set of commands to execute, and there is no relationship between sets.
When you run a job, HydraFlow will:

1. Generate all parameter combinations for the first set
2. Execute those commands
3. Move to the second set and generate its combinations
4. Execute those commands
5. And so on for each set

This allows you to organize different parameter sweep sets under the same job
configuration, sharing the same execution command (`run`, `call`, or `submit`).

### 3. Validate with Dry Run

Before executing, validate your job configuration with the `--dry-run` flag:

```bash
$ hydraflow run train --dry-run
```

This command displays the exact commands that would be executed without actually
running them, allowing you to verify parameter combinations and execution flow.

### 4. Execute the Job

Once validated, run the job with:

```bash
$ hydraflow run train
```

With the configuration above, this executes the following commands sequentially:

```bash
# First set commands
$ python train.py model=small learning_rate=0.1
$ python train.py model=large learning_rate=0.1
$ python train.py model=small learning_rate=0.2
$ python train.py model=large learning_rate=0.2

# Second set commands (completely independent from first set)
$ python train.py optimizer=adam dropout=0.1
$ python train.py optimizer=sgd dropout=0.1
$ python train.py optimizer=adam dropout=0.2
$ python train.py optimizer=sgd dropout=0.2
```

### 5. Parameter Types: `each`, `all`, and `add`

HydraFlow supports three types of parameters for configuring your commands:

- **`each`**: Creates a grid of parameter combinations, each resulting in a separate
  command. Parameters are expanded using the sweep syntax to generate multiple commands.

- **`all`**: Parameters included in every command from the set. These parameters
  can also use the sweep syntax, but all parameters are included as-is in each command.

- **`add`**: Additional arguments appended to the end of each command, primarily used
  for Hydra configuration. These are passed as-is, without any expansion.

Example of the three parameter types:

```yaml
jobs:
  train:
    run: python train.py
    add: hydra/launcher=joblib hydra.launcher.n_jobs=2  # Job-level add
    sets:
      - each: model=small,large  # Creates multiple commands
      - all: seed=42 debug=true  # Included in every command
      - add: hydra.job.num_nodes=1  # Appended to every command
```

This generates:

```bash
$ python train.py model=small seed=42 debug=true hydra/launcher=joblib hydra.launcher.n_jobs=2 hydra.job.num_nodes=1
$ python train.py model=large seed=42 debug=true hydra/launcher=joblib hydra.launcher.n_jobs=2 hydra.job.num_nodes=1
```

Note that the set-level `add` would override the job-level `add` if both were specified.

### 6. Parallelize with Submission Commands

The true power of HydraFlow's workflow management emerges when using job submission
commands like `sbatch` or `qsub`. Instead of running sequentially, you can submit
jobs to run in parallel on a cluster:

```yaml title="hydraflow.yaml"
jobs:
  train:
    submit: sbatch --partition=gpu --nodes=1 job.sh
    add: hydra/launcher=submitit_slurm
    sets:
      - each: >-
          model=small,large
          learning_rate=0.1,0.2
      - all: seed=42
```

This approach offers several advantages:
- **Parallelization**: Execute multiple parameter combinations simultaneously
- **Resource Optimization**: Allocate appropriate resources to each job
- **Scalability**: Easily scale to hundreds or thousands of experiments
- **Fault Isolation**: Failures in one job don't affect others

## When to Use Advanced Workflows

Consider using HydraFlow's advanced workflow features when:

- You need to explore complex parameter spaces with many configurations
- You want to organize multiple independent parameter sweeps under the same job
- You want to leverage distributed computing resources
- You need reproducible workflow definitions

## What's Next

The following pages explain HydraFlow's advanced features in detail:

- [Extended Sweep Syntax](sweep-syntax.md) - Learn how to define complex
  parameter spaces using HydraFlow's extended syntax

- [Job Configuration](job-configuration.md) - Define reusable and maintainable
  job configurations with sets

- [Batch Submission](batch-submission.md) - Run jobs efficiently on
  clusters and job schedulers