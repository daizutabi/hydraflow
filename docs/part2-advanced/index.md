# Automating Workflows

This section covers advanced techniques for automating and structuring
multiple experiments in HydraFlow. It provides tools for defining complex
parameter spaces and reusable experiment definitions.

## Overview

After creating your basic HydraFlow applications, the next step is to
automate your experiment workflows. This includes:

- Creating parameter sweeps across complex combinations
- Defining reusable experiment configurations
- Organizing large-scale experiment campaigns

## Key Components

The main components for workflow automation in HydraFlow are:

1. **Extended Sweep Syntax**: A powerful syntax for defining parameter
    spaces beyond simple comma-separated values.
2. **Job Configuration**: A YAML-based definition system for creating
    reusable experiment workflows.

## Practical Examples

For hands-on examples of workflow automation, see our
[Practical Tutorials](../practical-tutorials/index.md) section, specifically:

- [Automating Complex Workflows](../practical-tutorials/advanced.md): A tutorial
  that demonstrates how to use `hydraflow.yaml` to define and execute
  various types of workflows
- [Analyzing Experiment Results](../practical-tutorials/analysis.md): Learn
  how to work with results from automated experiment runs

## Extended Sweep Syntax

HydraFlow extends Hydra's sweep syntax to provide more powerful ways
to define parameter spaces:

```bash
# Range of values (inclusive)
python train.py -m "learning_rate=0.001:0.1:0.001"  # start:end:step

# SI prefixes
python train.py -m "batch_size=1k,2k,4k"  # 1000, 2000, 4000

# Logarithmic spacing
python train.py -m "learning_rate=log(0.0001:0.1:10)"  # 10 points log-spaced
```

Learn more about these capabilities in [Sweep Syntax](sweep-syntax.md).

## Job Configuration

For more complex experiment workflows, you can use HydraFlow's job
configuration system:

```yaml
jobs:
  train_models:
    run: python train.py
    sets:
      - each: model=small,medium,large
        all: seed=42 epochs=10

  evaluate_models:
    run: python evaluate.py
    sets:
      - each: model=small,medium,large
        all: test_data=validation
```

This approach allows you to define reusable experiment definitions that
can be executed with a single command. Learn more in
[Job Configuration](job-configuration.md).

## Executing Workflows

Once defined, workflows can be executed using the `hydraflow run` command:

```bash
# Execute a job defined in hydraflow.yaml
hydraflow run train_models

# Preview execution with dry run
hydraflow run train_models --dry-run

# Run a job with additional overrides
hydraflow run train_models seed=123
```

## What's Next

In the following pages, we'll explore workflow automation in detail:

- [Sweep Syntax](sweep-syntax.md): Learn about HydraFlow's extended
  syntax for defining parameter spaces.
- [Job Configuration](job-configuration.md): Discover how to create
  reusable job definitions for your experiments.

After automating your experiments, you'll want to analyze the results
using the tools covered in [Part 3: Analyzing Results](../part3-analysis/index.md).
