# Advanced Multi-Run Workflows

HydraFlow extends Hydra's capabilities with advanced features for efficient
multi-run workflows. While Hydra provides basic parameter sweeping, HydraFlow
offers tools for more complex scenarios, including extended sweep syntax
and reusable job definitions.

## Overview

Part 3 focuses on HydraFlow's advanced features for efficient multi-run execution:

- **Extended Sweep Syntax** - Define complex parameter spaces with numerical
  ranges, combinations, and engineering notation

- **Job Configuration** - Create reusable job definitions with GitHub
  Actions-like syntax

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
  for Hydra configuration. When specified at both job and set levels, they are merged
  with set-level values taking precedence for the same keys.

Example of the three parameter types:

```yaml
jobs:
  train:
    run: python train.py
    add: hydra/launcher=joblib hydra.launcher.n_jobs=2  # Job-level add
    sets:
      # First set - uses job-level add
      - each: model=small  # Creates a command
      - all: seed=42       # Included in the command

      # Second set - merges job-level add with set-level add
      - each: model=large
      - all: seed=43
      - add: hydra/launcher=submitit hydra.job.num_nodes=1  # Merges with job-level add
```

This generates:

```bash
# First set - uses job-level add
$ python train.py model=small seed=42 hydra/launcher=joblib hydra.launcher.n_jobs=2

# Second set - merges job-level add with set-level add (hydra/launcher is overridden)
$ python train.py model=large seed=43 hydra/launcher=submitit hydra.launcher.n_jobs=2 hydra.job.num_nodes=1
```

**Important**: When a set has its own `add` parameter, it is merged with the job-level `add`. If the same parameter key appears in both, the set-level value takes precedence. This allows you to define common parameters at the job level while customizing specific parameters for each set.

### 6. Batch Processing with Submit Command

HydraFlow's `submit` command provides a way to handle multiple parameter combinations as a batch:

```yaml title="hydraflow.yaml"
jobs:
  train:
    submit: python batch_processor.py
    sets:
      - each: >-
          model=small,large
          learning_rate=0.1,0.2
      - all: seed=42
```

Unlike the `run` command which executes once per parameter combination, the `submit` command:

1. Collects all parameter combinations from the sets
2. Writes these combinations to a text file (one combination per line)
3. Executes the specified command once, passing the text file as an argument

The key difference is:

- `run`: Each parameter combination triggers a separate command execution
- `submit`: All parameter combinations are passed to a single command execution

The custom handler script (e.g., `batch_processor.py` in the example) has complete freedom in how it processes the parameter file:

- It can read the parameter combinations and process them sequentially
- It can implement custom parallelization logic for local execution
- It can submit jobs to a cluster scheduler (like SLURM's sbatch)
- It can group parameter combinations for efficient resource utilization

For example, your handler script might:
```python
# Example batch_processor.py
import sys
import subprocess

# Read parameter combinations from the file passed as an argument
with open(sys.argv[1]) as f:
    param_combinations = f.read().splitlines()

# Process each combination however you want
for params in param_combinations:
    # Example: Submit to SLURM
    args = ["sbatch", "--job-name=train", "--wrap", f"python train.py {params}"]
    subprocess.run(args)

    # Or run locally in parallel
    # subprocess.Popen(f"python train.py {params}", shell=True)
```

This approach gives you complete flexibility in how parameters are processed and executed.

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