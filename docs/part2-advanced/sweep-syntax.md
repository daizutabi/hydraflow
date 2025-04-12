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

The format is `start:stop:step`, similar to Python's range
notation. **Note that unlike Python's range, the stop value
is inclusive** - the range includes both the start and stop
values if they align with the step size.

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
          weight_decay=1:3:n     # nano (1e-9)
          max_tokens=1:3:k       # kilo (1e3)
          model_dim=1:3:M        # mega (1e6)
```

This generates all combinations (total of 27 different commands).

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
          batch_size/k=4,8        # Applies kilo (1e3) to all values
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
          model=(cnn,transformer)_(small,large)  # Combines model types and sizes
```

This generates:

```bash
python train.py -m model=cnn_small
python train.py -m model=cnn_large
python train.py -m model=transformer_small
python train.py -m model=transformer_large
```

Parentheses are particularly useful for combining values with SI prefixes:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          weight_decay=(1,4)k,(6,8)M  # Combines values with different prefixes
```

## Pipe Operator for Multiple Parameter Sets

The pipe operator (`|`) allows you to specify completely
different parameter sets that are executed independently:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: >-
          model=small,large|lr=0.1,0.2|dropout=1:5:2 decay=1,2
```

This generates separate Hydra multirun commands:

```bash
# The pipe operator creates separate Hydra multirun commands
python train.py -m model=small,large decay=1
python train.py -m model=small,large decay=2
python train.py -m lr=0.1,0.2 decay=1
python train.py -m lr=0.1,0.2 decay=2
python train.py -m dropout=1,3,5 decay=1
python train.py -m dropout=1,3,5 decay=2
```

The pipe operator splits the expression into separate
Hydra multirun commands.
Each section before a pipe becomes a separate command
with its own grid sweep.
Parameters that appear after all pipes
(like `decay=1,2` in the example) are applied to
every section and expanded as well.

This is fundamentally different from `each` without
pipes, which would create a single grid of all combinations.
The pipe operator allows you to run completely independent
parameter sweeps in the same job.

A practical use case is to group similar configurations
while separating dissimilar ones:

```yaml
jobs:
  train:
    run: python train.py
    sets:
      - each: model=small,medium|large
```

This is equivalent to `model=small,medium|model=large` and
would generate two separate command sets:

```bash
python train.py -m model=small,medium
python train.py -m model=large
```

This allows you to group smaller models (small and medium) in
one job while running the large model in a separate job,
which is useful when you want to allocate resources
differently based on model size.

## Summary

HydraFlow's extended sweep syntax provides several powerful features for parameter space exploration:

1. **Basic comma-separated lists** - Simple way to enumerate discrete parameter values
2. **Numerical ranges** - Define continuous ranges with start:stop:step notation (inclusive of stop value)
3. **SI prefixes** - Use scientific notation shortcuts (n, u, m, k, M, G, etc.) for large/small numbers
4. **Prefix notation** - Apply SI prefixes to all values in a parameter list
5. **Parentheses grouping** - Create combinations of values and nested structures
6. **Pipe operator** - Run multiple independent parameter sweeps in the same job

All of these can be combined to create complex, expressive parameter sweeps with minimal configuration. Remember that using the `each` keyword creates a cartesian product of all parameters (all possible combinations), while the pipe operator (`|`) creates separate, independent parameter sweeps.

When using these features, HydraFlow will automatically generate the appropriate Hydra multirun commands with the `-m` flag.
