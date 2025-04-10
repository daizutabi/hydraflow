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
      - batch: >-
          model=small,large
          learning_rate=0.1,0.01
```

### Configuration Structure

The configuration file uses the following structure:

- `jobs`: The top-level key containing all job definitions
  - `<job_name>`: Name of the job (e.g., "train")
    - `run`: The command to execute
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
      - batch: model=small,large
```

### `call`

The `call` command executes a Python function:

```yaml
jobs:
  train:
    call: my_module.train_function
    sets:
      - batch: model=small,large
```

The specified function will be imported and called with the parameters.

### `submit`

The `submit` command is used for submitting jobs to a cluster scheduler:

```yaml
jobs:
  train:
    submit: sbatch --partition=gpu job.sh
    sets:
      - batch: model=small,large
```

The parameters will be passed to the submit command.

## Parameter Sets

Each job contains one or more parameter sets under the `sets` key.
Each set can include the following types of parameters:

### `batch`

The `batch` parameter defines a grid of parameter combinations. Each combination
will be executed as a separate command:

```yaml
sets:
  - batch: >-
      model=small,large
      learning_rate=0.1,0.01
```

This will generate four separate executions, one for each combination of
model and learning rate.

### `args`

The `args` parameter defines parameters that will be included in each
execution from the set:

```yaml
sets:
  - batch: model=small,large
  - args: seed=42 debug=true
```

This will include `seed=42 debug=true` in every execution for the set.

### `with`

The `with` parameter adds environment variables to the execution:

```yaml
sets:
  - batch: model=small,large
  - with: >-
      CUDA_VISIBLE_DEVICES=0
      OMP_NUM_THREADS=4
```

Environment variables are set before executing the command.

## Multiple Parameter Sets

A job can have multiple parameter sets, each executed independently:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      # First set: Train models with different architectures
      - batch: >-
          model=small,large
          optimizer=adam

      # Second set: Train models with different learning rates
      - batch: >-
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
    sets:
      - batch: model=small,large
      - args: seed=42 debug=true
      - with: CUDA_VISIBLE_DEVICES=0
```

This will execute:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py model=small seed=42 debug=true
CUDA_VISIBLE_DEVICES=0 python train.py model=large seed=42 debug=true
```

## Extended Sweep Syntax

HydraFlow supports an extended syntax for defining parameter sweeps:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - batch: >-
          model=small,large
          learning_rate=logspace(0.1,0.001,4)
          weight_decay=0,1e-4,1e-5
```

This uses HydraFlow's extended sweep syntax to generate a grid of
parameters. See [Extended Sweep Syntax](sweep-syntax.md) for details.

## Job Inheritance

You can define a base job and inherit from it:

```yaml
jobs:
  base_train:
    run: python train.py
    sets:
      - args: batch_size=32 seed=42

  finetune:
    run: python train.py
    inherit: base_train
    sets:
      - batch: model=small,large
      - args: learning_rate=0.0001
```

The `finetune` job inherits the `run` command and the parameter sets from
`base_train`, and adds its own parameter sets.

## Overriding Inherited Sets

When inheriting, you can override specific elements:

```yaml
jobs:
  base_train:
    run: python train.py
    sets:
      - args: batch_size=32 seed=42

  finetune:
    # Override the command
    run: python finetune.py
    inherit: base_train
    sets:
      # Add new parameter sets
      - batch: model=small,large
```

## Inheritance Chains

You can create chains of inheritance:

```yaml
jobs:
  base:
    run: python train.py
    sets:
      - args: batch_size=32

  train:
    inherit: base
    sets:
      - args: seed=42

  finetune:
    inherit: train
    sets:
      - batch: model=small,large
```

## Using Variables

You can define variables at the top level and use them in jobs:

```yaml
variables:
  base_lr: 0.001
  models: small,large

jobs:
  train:
    run: python train.py
    sets:
      - batch: >-
          model=${models}
          learning_rate=${base_lr}
```

Variables are interpolated when the configuration is loaded.

## Conditional Sets

You can conditionally include sets based on environment variables:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - batch: model=small,large

      # Only included if DEBUG=1 is set
      - if: ${env:DEBUG} == 1
        args: debug=true verbose=true
```

The condition is evaluated when the job is executed.

## Including External Configurations

You can include external configuration files:

```yaml
# Main hydraflow.yaml
include:
  - path: configs/base.yaml
  - path: configs/custom.yaml

jobs:
  # Local job definitions
  custom_job:
    run: python custom.py
```

The included files should have the same structure as the main file.

## Reusing Sets

You can define reusable parameter sets and reference them:

```yaml
param_sets:
  common:
    args: seed=42 batch_size=32

  models:
    batch: model=small,large

jobs:
  train:
    run: python train.py
    sets:
      - use: common
      - use: models
```

This allows you to define parameter sets once and reuse them across
multiple jobs.

## Complete Example

Here's a complete example that demonstrates many of the features:

```yaml
variables:
  base_lr: 0.001
  models: small,large,xlarge

param_sets:
  common:
    args: seed=42 batch_size=32 epochs=100

  optimization:
    batch: >-
      optimizer=adam,sgd
      learning_rate=${base_lr},${base_lr}*0.1

jobs:
  # Base job definition
  base_train:
    run: python train.py
    sets:
      - use: common
      - with: CUDA_VISIBLE_DEVICES=0

  # Training job
  train:
    inherit: base_train
    sets:
      - batch: model=${models}
      - use: optimization

  # Evaluation job
  evaluate:
    run: python evaluate.py
    sets:
      - use: common
      - batch: model=${models}
      - args: eval_split=test

  # Fine-tuning job
  finetune:
    inherit: train
    run: python finetune.py
    sets:
      - args: pretrained=true learning_rate=${base_lr}*0.01
```

## Best Practices

### Organize Related Parameters

Group related parameters together in the same set:

```yaml
sets:
  # Model architecture parameters
  - batch: >-
      model=small,large
      num_layers=2,4,6

  # Optimization parameters
  - batch: >-
      optimizer=adam,sgd
      learning_rate=0.1,0.01
```

### Use Descriptive Job Names

Choose descriptive names for your jobs:

```yaml
jobs:
  pretrain_transformer:
    # ...

  finetune_classification:
    # ...

  evaluate_test_set:
    # ...
```

### Validate with Dry Run

Always validate your job configuration with a dry run before executing:

```bash
$ hydraflow run train --dry-run
```

### Use Variables for Shared Values

Define variables for values used across multiple jobs:

```yaml
variables:
  base_epochs: 100
  base_batch_size: 32

jobs:
  train:
    # ...
    sets:
      - args: epochs=${base_epochs} batch_size=${base_batch_size}
```

### Document Your Configurations

Add comments to explain complex configurations:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      # These parameters control the model architecture
      - batch: >-
          model=small,large  # Model size
          num_layers=2,4     # Number of transformer layers

      # These control the optimization process
      - batch: >-
          optimizer=adam,sgd  # Optimization algorithm
```

### Version Control Your Job Configurations

Keep your `hydraflow.yaml` files in version control to ensure
reproducibility of your experiments.