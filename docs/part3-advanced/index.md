# Advanced Multi-Run Workflows

HydraFlow extends Hydra's capabilities with advanced features for efficient
multi-run workflows. While Hydra provides basic parameter sweeping, HydraFlow
offers tools for more complex scenarios, including extended sweep syntax
and reusable job definitions.

## Overview

Part 3 focuses on HydraFlow's advanced features for scaling and automating
your experiment workflows:

- **Extended Sweep Syntax** - Define complex parameter spaces with numerical
  ranges, combinations, and engineering notation
- **Job Configuration** - Create reusable job definitions with GitHub
  Actions-like syntax

These advanced features build upon the basics covered in [Part 1: Running Applications](../part1-applications/index.md)
and work seamlessly with the analysis capabilities from [Part 2: Analyzing Results](../part2-analysis/index.md).

## Extended Sweep Syntax

HydraFlow's extended sweep syntax allows you to define complex parameter spaces
more concisely than Hydra's basic comma-separated lists:

```yaml
# Define numerical ranges with colons
batch_size=16:128:16  # From 16 to 128 in steps of 16

# Use SI prefixes for large/small numbers
learning_rate=1:5:m   # 1e-3 to 5e-3

# Define combinations with parentheses
model=(cnn,transformer)_(small,large)
```

This powerful syntax can dramatically reduce the verbosity of parameter sweeps
while making them more readable and maintainable.

[Learn more about Extended Sweep Syntax](sweep-syntax.md)

## Job Configuration

HydraFlow's job configuration system allows you to define reusable experiment
definitions in YAML format:

```yaml
# hydraflow.yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: model=small,large
      - all: seed=42
```

Key concepts in job configuration:

- **Execution Commands**: Choose between `run`, `call`, or `submit` for different execution modes
- **Parameter Sets**: Define independent parameter sets with `each`, `all`, and `add` parameters
- **Job Organization**: Group related parameters and reuse configurations

Using the CLI, you can execute defined jobs:

```bash
$ hydraflow run train
```

[Learn more about Job Configuration](job-configuration.md)

## Integration with HydraFlow Ecosystem

These advanced features integrate with the rest of the HydraFlow ecosystem:

1. **Configuration** → **Execution** → **Analysis** workflow:
   - Define configurations in Python dataclasses ([Part 1](../part1-applications/configuration.md))
   - Execute experiments with advanced sweep syntax and job definitions ([Part 3](job-configuration.md))
   - Analyze results with structured APIs ([Part 2](../part2-analysis/run-collection.md))

2. **Type-safety throughout**:
   - Configuration is type-checked via dataclasses
   - Parameters are validated during job execution
   - Results can be analyzed with type-aware APIs

## When to Use Advanced Workflows

Consider using HydraFlow's advanced workflow features when:

- You need to explore complex parameter spaces with many configurations
- You want to organize multiple independent parameter sweeps under the same job
- You need reproducible workflow definitions that can be version-controlled
- You're running large-scale experiments that benefit from automation

## What's Next

The following pages explain HydraFlow's advanced features in detail:

- [Extended Sweep Syntax](sweep-syntax.md) - Learn how to define complex
  parameter spaces using HydraFlow's extended syntax

- [Job Configuration](job-configuration.md) - Define reusable and maintainable
  job configurations with sets