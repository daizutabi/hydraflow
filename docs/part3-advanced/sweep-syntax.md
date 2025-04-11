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
python train.py -m model=small
python train.py -m model=medium
python train.py -m model=large
```

When using multiple parameters with `each`, all possible combinations (cartesian product) will be generated:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          model=small,medium
          learning_rate=0.1,0.01
```

This generates all four combinations:

```bash
python train.py -m model=small learning_rate=0.1
python train.py -m model=small learning_rate=0.01
python train.py -m model=medium learning_rate=0.1
python train.py -m model=medium learning_rate=0.01
```

## Numerical Ranges

For numerical parameters, you can use range notation with colons:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: batch_size=16:128:16
```

This generates:

```bash
python train.py -m batch_size=16
python train.py -m batch_size=32
python train.py -m batch_size=48
...
python train.py -m batch_size=128
```

The format is `start:stop:step`, similar to Python's range notation. **Note that unlike Python's range, the stop value is inclusive** - the range includes both the start and stop values if they align with the step size.

You can omit the start value to default to 0:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: steps=:5  # Equivalent to steps=0:5:1
```

Generates:

```bash
python train.py -m steps=0
python train.py -m steps=1
python train.py -m steps=2
python train.py -m steps=3
python train.py -m steps=4
python train.py -m steps=5
```

You can also use negative steps to create descending ranges:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: lr=5:1:-1
```

Generates:

```bash
python train.py -m lr=5
python train.py -m lr=4
python train.py -m lr=3
python train.py -m lr=2
python train.py -m lr=1
```

## SI Prefixes (Engineering Notation)

You can use SI prefixes to represent large or small numbers concisely:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          learning_rate=1:5:m    # milli (1e-3)
          weight_decay=1:3:n     # nano (1e-9)
          max_tokens=1:3:k       # kilo (1e3)
          model_dim=1:3:M        # mega (1e6)
```

This generates all combinations (total of 81 commands):

```bash
python train.py -m learning_rate=1e-3 weight_decay=1e-9 max_tokens=1e3 model_dim=1e6
python train.py -m learning_rate=1e-3 weight_decay=1e-9 max_tokens=1e3 model_dim=2e6
python train.py -m learning_rate=1e-3 weight_decay=1e-9 max_tokens=1e3 model_dim=3e6
python train.py -m learning_rate=1e-3 weight_decay=1e-9 max_tokens=2e3 model_dim=1e6
...
python train.py -m learning_rate=5e-3 weight_decay=3e-9 max_tokens=3e3 model_dim=3e6
```

Note: The `each` parameter creates a grid of all possible combinations (cartesian product) of the parameter values. The example above would generate 3×3×3×3=81 different commands in total.

Supported SI prefixes:
- `f`: femto (1e-15)
- `p`: pico (1e-12)
- `n`: nano (1e-9)
- `u`: micro (1e-6)
- `m`: milli (1e-3)
- `k`: kilo (1e3)
- `M`: mega (1e6)
- `G`: giga (1e9)
- `T`: tera (1e12)

You can also use fractional steps with SI prefixes:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: learning_rate=0.1:0.4:0.1:m  # From 0.1e-3 to 0.4e-3 by 0.1e-3
```

## Prefix Notation

You can apply an SI prefix to all values in a parameter using the prefix notation:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          lr/m=1,2,5,10           # Applies milli (1e-3) to all values
          batch_size/k=4,8,16,32  # Applies kilo (1e3) to all values
```

This generates:

```bash
python train.py -m lr=1e-3 batch_size=4e3
python train.py -m lr=2e-3 batch_size=8e3
...
```

This is useful when all values for a parameter share the same exponent.

## Grouping with Parentheses

You can use parentheses to create combinations of values:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          dropout=(0.1,0.2,0.3)(small,large)  # Combines all values
```

This generates:

```bash
python train.py -m dropout=0.1small
python train.py -m dropout=0.2small
python train.py -m dropout=0.3small
python train.py -m dropout=0.1large
python train.py -m dropout=0.2large
python train.py -m dropout=0.3large
```

You can also combine parentheses with SI prefixes:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: learning_rate=(1:3,5:7:2)m  # 1e-3, 2e-3, 3e-3, 5e-3, 7e-3
```

Or use them with exponents:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: learning_rate=(1,2)e(-1,-2,-3)  # 1e-1, 2e-1, 1e-2, 2e-2, 1e-3, 2e-3
```

## Pipe Operator for Multiple Parameter Sets

The pipe operator (`|`) allows you to specify completely different parameter sets that don't form a grid:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          model=small learning_rate=0.1|model=medium learning_rate=0.01|model=large learning_rate=0.001
```

This generates exactly these three commands (unlike `each` which would create a grid of 9 combinations):

```bash
python train.py -m model=small learning_rate=0.1
python train.py -m model=medium learning_rate=0.01
python train.py -m model=large learning_rate=0.001
```

The pipe operator is useful when you want to create specific parameter combinations rather than a full grid search. It lets you precisely control which combinations to run, unlike `each` without pipes which generates all possible combinations.

You can continue parameter specifications after a pipe by omitting the parameter name:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: model=small|medium|large
```

This is equivalent to `model=small|model=medium|model=large`.

## Combining Multiple Features

You can combine all these features for complex parameter sweeps:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          model/type=transformer,lstm|cnn,gru
          learning_rate/m=1:5:1|6:10:2
          dropout=(0.1:0.5:0.1)(before,after)
          batch_size/k=1,2,4,8
```

This creates a sophisticated parameter space with specific combinations of models, learning rates, dropout values and positions, and batch sizes.

## Practical Examples

### Learning Rate Sweep with SI Prefixes

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: learning_rate/m=1,5,10,50,100  # 1e-3, 5e-3, 10e-3, 50e-3, 100e-3
```

### Model Size and Decoder Layers Grid

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          model.size=small,medium,large
          model.decoder_layers=2:12:2  # 2, 4, 6, 8, 10, 12
```

### Targeted Combinations without Full Grid

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          model=small lr/m=10|model=medium lr/m=5|model=large lr/m=1
```

### Exponential Decay Rates

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: decay_rate=0.9,0.99,0.999,0.9999
```

### Combined Hyperparameters

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          optimizer=(adam,sgd)(lr/m=1,10)
```

This creates combinations like `optimizer=adamlr=1e-3`, `optimizer=adamlr=10e-3`, etc.

Remember to use `--dry-run` to verify your parameter sweeps before execution:

```bash
$ hydraflow run train --dry-run
```