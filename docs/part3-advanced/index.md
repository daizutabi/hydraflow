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

```yaml
# 1. Define a job in hydraflow.yaml
jobs:
  train:
    run: python train.py
    steps:
      # Each combination creates a separate execution (sequential unless using 'submit')
      - batch: >-
          model=transformer,lstm
          learning_rate=0.1,0.01,0.001

      # Parameters that will be expanded but included in each batch execution
      - args: optimizer=adam,sgd  # Expanded but included with each batch command
```

```bash
# 2. Run the job using the CLI
$ hydraflow run train

# 3. Or perform a dry run to see expanded commands
$ hydraflow run train --dry-run

# 4. Submit to a cluster
$ hydraflow run train
```

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