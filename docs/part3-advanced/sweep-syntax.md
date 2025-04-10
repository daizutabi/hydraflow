# Extended Sweep Syntax

HydraFlow provides an extended syntax for defining parameter spaces. This
syntax is more powerful than Hydra's basic comma-separated lists, allowing
you to define ranges, logarithmic spaces, and more.

## Basic Syntax

The core of HydraFlow's sweep syntax is a comma-separated list:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: model=small,medium,large
```

This generates commands for each parameter value:

```bash
python train.py model=small
python train.py model=medium
python train.py model=large
```

## Numerical Ranges

For numerical parameters, you can use range notation:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: batch_size=16:128:16
```

This generates:

```bash
python train.py batch_size=16
python train.py batch_size=32
python train.py batch_size=48
...
python train.py batch_size=128
```

The format is `start:stop:step`, similar to Python's range notation.

## Logarithmic Ranges

For learning rates and other parameters that benefit from logarithmic spacing,
use the `logspace` function:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          learning_rate=logspace(0.0001,0.1,5)
```

This generates logarithmically spaced values:

```bash
python train.py learning_rate=0.0001
python train.py learning_rate=0.001
python train.py learning_rate=0.01
python train.py learning_rate=0.1
```

## Linear Ranges

For linearly spaced values, use the `linspace` function:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          dropout=linspace(0.0,0.5,6)
```

This generates evenly spaced values:

```bash
python train.py dropout=0.0
python train.py dropout=0.1
python train.py dropout=0.2
python train.py dropout=0.3
python train.py dropout=0.4
python train.py dropout=0.5
```

## Scientific Notation

You can use scientific notation for very small or large numbers:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          weight_decay=0,1e-4,1e-5,1e-6
```

This generates:

```bash
python train.py weight_decay=0
python train.py weight_decay=0.0001
python train.py weight_decay=0.00001
python train.py weight_decay=0.000001
```

## Combining Multiple Parameters

You can combine multiple parameters in a single each:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          model=small,large
          learning_rate=0.1,0.01,0.001
```

This generates a grid of all combinations:

```bash
python train.py model=small learning_rate=0.1
python train.py model=small learning_rate=0.01
python train.py model=small learning_rate=0.001
python train.py model=large learning_rate=0.1
python train.py model=large learning_rate=0.01
python train.py model=large learning_rate=0.001
```

## Mixed Parameter Types

You can mix different parameter types and notations:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          model=small,large
          learning_rate=logspace(0.0001,0.1,4)
          dropout=0.0:0.5:0.1
```

This combines all parameter types in a single grid.

## Nested Parameters

You can sweep over nested parameters using dot notation:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          model.type=cnn,transformer
          model.layers=2,4,8
```

This generates:

```bash
python train.py model.type=cnn model.layers=2
python train.py model.type=cnn model.layers=4
python train.py model.type=cnn model.layers=8
python train.py model.type=transformer model.layers=2
python train.py model.type=transformer model.layers=4
python train.py model.type=transformer model.layers=8
```

## Boolean Parameters

Boolean parameters can be specified:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: use_augmentation=true,false
```

This generates:

```bash
python train.py use_augmentation=true
python train.py use_augmentation=false
```

## Expressions and Calculations

You can use simple mathematical expressions:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          learning_rate=0.01,0.01*0.1,0.01*0.01
```

This generates:

```bash
python train.py learning_rate=0.01
python train.py learning_rate=0.001
python train.py learning_rate=0.0001
```

## Using Variables

You can define variables and use them in parameter sweeps:

```yaml
variables:
  base_lr: 0.01
  models: small,large

jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          model=${models}
          learning_rate=${base_lr},${base_lr}*0.1
```

Variables are expanded when the configuration is loaded.

## Understanding `each`, `all`, and `add`

HydraFlow uses three different parameter types:

1. **`each`**: Creates separate executions for each parameter combination.
2. **`all`**: Parameters included in every command but can still use sweep syntax.
3. **`add`**: Additional arguments appended as-is to each command (no sweep expansion).

Example with all three types:

```yaml
jobs:
  train:
    run: python train.py
    add: hydra/launcher=joblib hydra.launcher.n_jobs=2
    sets:
      - each: model=small,large
      - all: seed=42,43 epochs=100
      - add: hydra.job.num_nodes=1
```

This generates:

```bash
python train.py model=small seed=42,43 epochs=100 hydra/launcher=joblib hydra.launcher.n_jobs=2 hydra.job.num_nodes=1
python train.py model=large seed=42,43 epochs=100 hydra/launcher=joblib hydra.launcher.n_jobs=2 hydra.job.num_nodes=1
```

Note that `seed=42,43` from `all` is passed as-is to the command but isn't expanded,
whereas `model=small,large` from `each` creates separate commands.

## Advanced Examples

### Complex Parameter Grid

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          model=resnet18,resnet50
          optimizer=adam,sgd
          learning_rate=logspace(0.0001,0.1,3)
          batch_size=16:128:16
```

### Specialized Configuration Sets

```yaml
jobs:
  train:
    run: python train.py
    sets:
      # Sweep over model architectures
      - each: >-
          model.type=cnn,transformer,mlp
          model.size=small,medium,large

      # Sweep over optimization parameters
      - each: >-
          optimizer=adam,sgd,adagrad
          learning_rate=logspace(0.0001,0.1,4)
```

### Conditional Parameters

You can create conditional parameters using multiple sets:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      # CNN-specific parameters
      - each: model.type=cnn
      - all: model.kernel_size=3,5,7

      # Transformer-specific parameters
      - each: model.type=transformer
      - all: model.num_heads=4,8
```

## Best Practices

### Keep Sets Manageable

While it's possible to create very large parameter spaces, try to keep them
manageable:

```yaml
# Instead of this (creates 1000 combinations)
sets:
  - each: >-
      param1=1:10:1
      param2=1:10:1
      param3=1:10:1

# Consider this (creates 30 combinations)
sets:
  - each: param1=1:10:1
  - each: param2=1:10:1
  - each: param3=1:10:1
```

### Use Descriptive Parameter Names

Choose descriptive parameter names to make your sweeps more understandable:

```yaml
# Less clear
sets:
  - each: >-
      lr=0.1,0.01
      bs=32,64

# More clear
sets:
  - each: >-
      learning_rate=0.1,0.01
      batch_size=32,64
```

### Document Parameter Ranges

Add comments to explain parameter ranges and their significance:

```yaml
sets:
  # Exploring learning rate impact on convergence
  - each: >-
      # Range covers typical values for Adam optimizer
      learning_rate=logspace(0.0001,0.01,5)
```

### Validate Your Sweeps

Always validate your parameter sweeps with a dry run before executing:

```bash
$ hydraflow run train --dry-run
```