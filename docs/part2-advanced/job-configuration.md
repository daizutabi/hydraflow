# Job Configuration

HydraFlow job configuration allows you to define reusable experiment
definitions that can be executed with a single command. This page explains
how to create and use job configurations.

## Basic Job Configuration

HydraFlow reads job definitions from a `hydraflow.yaml` file in your
project directory. A basic job configuration looks like this:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          model=small,large
          learning_rate=0.1,0.01
```

### Configuration Structure

The configuration file uses the following structure:

- `jobs`: The top-level key containing all job definitions
  - `<job_name>`: Name of the job (e.g., "train")
    - `run`: The command to execute
    - `add`: Global configuration arguments appended to each command
    - `sets`: List of parameter sets for the job

Each job must have either a `run`, `call`, or `submit` key, and at least one
parameter set.

## Execution Commands

HydraFlow supports three types of execution commands:

### `run`

The `run` command executes the specified command directly:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: model=small,large
```

### `call`

The `call` command executes a Python function:

```yaml
jobs:
  train:
    call: my_module.train_function
    sets:
      - each: model=small,large
```

The specified function will be imported and called with the parameters.

### `submit`

The `submit` command collects all parameter combinations into a text
file and passes this file to the specified command:

```yaml
jobs:
  train:
    submit: python submit_handler.py
    sets:
      - each: model=small,large
```

When executed, this will:

1. Generate all parameter combinations from the sets
2. Write these combinations to a text file (one combination per line)
3. Execute the specified command once, passing the text file as an argument

The command (e.g., `submit_handler.py` in the example) is responsible for:

1. Reading the parameter file
2. Processing the parameter sets in any way it chooses
3. Optionally distributing the work (via cluster jobs, local parallelization, etc.)

The key difference between `run` and `submit`:

- `run`: Executes the command once per parameter combination
- `submit`: Executes the command once, with all parameter combinations provided in a file

This gives you complete flexibility in how parameter combinations are
processed. Your handler script can implement any logic - from simple
sequential processing to complex distributed execution across a cluster.

## Parameter Sets

Each job contains one or more parameter sets under the `sets` key.
Each set can include the following types of parameters:

### `each`

The `each` parameter defines a grid of parameter combinations. Each combination
will be executed as a separate command:

```yaml
sets:
  - each: >-
      model=small,large
      learning_rate=0.1,0.01
```

This will generate four separate executions, one for each combination of
model and learning rate.

### `all`

The `all` parameter defines parameters that will be included in each
execution from the set:

```yaml
sets:
  - each: model=small,large
  - all: seed=42 debug=true
```

This will include `seed=42 debug=true` in every execution for the set.

### `add`

The `add` parameter adds additional arguments that are appended to the end
of each command. This is primarily used for Hydra configuration settings:

```yaml
sets:
  - each: model=small,large
  - add: >-
      hydra/launcher=joblib
      hydra.launcher.n_jobs=4
```

This will append Hydra configuration to each command from the set.
If a set has its own `add` parameter, it completely overrides the job-level `add` parameter
(they are not merged). The job-level `add` is entirely ignored for that set.

## Multiple Parameter Sets

A job can have multiple parameter sets, each executed independently:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      # First set: Train models with different architectures
      - each: >-
          model=small,large
          optimizer=adam

      # Second set: Train models with different learning rates
      - each: >-
          model=medium
          learning_rate=0.1,0.01,0.001
```

Each set is completely independent and does not build upon the others.
The sets are executed sequentially in the order they are defined.

## Combining Parameter Types

You can combine different parameter types within a single set:

```yaml
jobs:
  train:
    run: python train.py
    add: hydra/launcher=joblib hydra.launcher.n_jobs=2
    sets:
      # First set: uses job-level add
      - each: model=small
      - all: seed=42 debug=true

      # Second set: merges with job-level add (set-level parameters take precedence)
      - each: model=large
      - all: seed=43
      - add: hydra/launcher=submitit hydra.launcher.submitit.cpus_per_task=4
```

This will execute:

```bash
# First set: with job-level add
python train.py model=small seed=42 debug=true hydra/launcher=joblib hydra.launcher.n_jobs=2

# Second set: merges job-level and set-level add (hydra/launcher is overridden by set-level)
python train.py model=large seed=43 hydra/launcher=submitit hydra.launcher.n_jobs=2 hydra.launcher.submitit.cpus_per_task=4
```

## Job-level and Set-level `add`

You can specify `add` at both the job level and set level:

```yaml
jobs:
  train:
    run: python train.py
    add: hydra/launcher=joblib hydra.launcher.n_jobs=2
    sets:
      # Uses job-level add
      - each: model=small,medium

      # Merges with job-level add (set-level takes precedence for the same keys)
      - each: model=large,xlarge
        add: hydra/launcher=submitit hydra.launcher.submitit.cpus_per_task=8
```

When a set has its own `add` parameter, it is merged with
the job-level `add` parameter.
If the same parameter key exists in both the job-level and set-level
`add`, the set-level value takes precedence.

For example, with the configuration above:

- The first set uses: `hydra/launcher=joblib hydra.launcher.n_jobs=2`
- The second set uses: `hydra/launcher=submitit hydra.launcher.n_jobs=2 hydra.launcher.submitit.cpus_per_task=8`

Notice how `hydra/launcher` is overridden by the set-level value,
while `hydra.launcher.n_jobs` from the job-level is retained.

This behavior allows you to:

1. Define common parameters at the job level
2. Override or add specific parameters at the set level
3. Keep all non-conflicting parameters from both levels

This merging behavior makes it easy to maintain common configuration
options while customizing specific aspects for different parameter sets.

## Summary

HydraFlow's job configuration system provides a powerful way to define
and manage complex parameter sweeps:

1. **Execution Commands**:

    - `run`: Executes a command once per parameter combination (most common usage)
    - `call`: Calls a Python function once per parameter combination
    - `submit`: Passes all parameter combinations as a text file to a handler script, executed once

2. **Parameter Types**:

    - `each`: Generates a grid of parameter combinations (cartesian product)
    - `all`: Specifies parameters included in every command
    - `add`: Arguments appended to the end of each command (primarily for Hydra configuration)

3. **Multiple Sets and Merging Behavior**:

    - Define multiple independent parameter sets
    - Job-level and set-level `add` parameters are merged
    - Set-level values take precedence for the same keys

These features combined allow you to define complex
experiment configurations concisely and execute
them efficiently. Reusing configurations ensures
reproducibility and consistency across your experiments.
