# Extended Sweep Syntax

HydraFlow extends Hydra's parameter sweep capabilities with a powerful and
flexible syntax. This page explains HydraFlow's extended syntax for defining
complex parameter spaces.

## Basic Concepts

While Hydra's built-in syntax allows for basic parameter sweeps like:

```bash
python train.py -m learning_rate=0.1,0.01,0.001
```

HydraFlow provides a much richer syntax, including:

- **Numerical ranges** with start, stop, and step values
- **Engineering notation** support
- **Nested syntax** for complex combinations
- **Automatic expansion** of parameter spaces

## Numerical Ranges

HydraFlow supports defining numerical ranges using a colon-separated syntax:

```
# Format: start:stop (uses step=1 for integers)
1:5        # Expands to 1, 2, 3, 4, 5

# Format: start:step:stop
1:0.5:3    # Expands to 1, 1.5, 2, 2.5, 3

# Decreasing ranges (requires negative step)
5:-1:1     # Expands to 5, 4, 3, 2, 1

# For learning rates (decreasing from 0.1 to 0.001)
0.1:-0.01:0.001  # Decreasing by 0.01 steps: 0.1, 0.09, 0.08, ..., 0.001
# Or more commonly, use comma notation for log-scale:
0.1,0.01,0.001   # Log-scale decrease (clearer for order-of-magnitude changes)
```

This is especially useful for defining learning rate schedules,
regularization strengths, or any parameter that needs systematic exploration.

## Engineering Notation

HydraFlow supports engineering notation for scientific values:

```
# With suffixes
1M         # 1,000,000 (1e6)
1k         # 1,000 (1e3)
10m        # 0.01 (1e-2)
1u         # 0.000001 (1e-6)

# In ranges
1m:10m     # 0.001 to 0.01
100k:1M    # 100,000 to 1,000,000
```

Supported suffixes include:
- `T` (tera, 1e12)
- `G` (giga, 1e9)
- `M` (mega, 1e6)
- `k` (kilo, 1e3)
- `m` (milli, 1e-3)
- `u` (micro, 1e-6)
- `n` (nano, 1e-9)
- `p` (pico, 1e-12)
- `f` (femto, 1e-15)

## List Syntax

Define lists of values with commas:

```
# Simple values
model=cnn,transformer,lstm    # Three model types

# Combined with ranges
lr=0.1,0.01,0.001,0.0001     # Explicit values for clarity
```

## Parentheses for Grouping

Use parentheses to group related values:

```
# Group related values
optimizer=(sgd,0.1),(adam,0.01)

# Expands to:
#   optimizer=sgd learning_rate=0.1
#   optimizer=adam learning_rate=0.01
```

## Combined Parameters

Define combinations of parameters:

```
# Combined parameter sweep
python train.py --multirun "model=cnn,transformer batch_size=32,64,128"
```

The above command expands to:

```
model=cnn batch_size=32
model=cnn batch_size=64
model=cnn batch_size=128
model=transformer batch_size=32
model=transformer batch_size=64
model=transformer batch_size=128
```

## Advanced Examples

### Logarithmic Sweeps

```
# Logarithmic sweep of learning rates (comma notation is clearer)
lr=1,0.1,0.01,0.001,0.0001  # Explicit log-scale values

# Using range notation for regular intervals
lr=0.001:0.001:0.01  # Increasing: 0.001, 0.002, 0.003, ..., 0.01
lr=0.01:-0.001:0.001 # Decreasing: 0.01, 0.009, 0.008, ..., 0.001
```

### Grid Search with Multiple Parameters

```
# Grid search
"model=cnn,transformer,lstm lr=0.1,0.01,0.001 batch_size=32,64"
```

This expands to all combinations (18 total runs):
- 3 models × 3 learning rates × 2 batch sizes

### Complex Grouped Parameters

```
# Complex groups with optimization parameters
"optimizer=(sgd,0.9,0.1),(adam,0.99,0.01)"
```

Expands to:
```
optimizer=sgd momentum=0.9 learning_rate=0.1
optimizer=adam beta=0.99 learning_rate=0.01
```

## Using the Syntax in Jobs

You can use the extended syntax directly in your job definitions:

```yaml
jobs:
  train:
    run: python train.py
    steps:
      - batch: >-
          model=cnn,transformer,lstm
          learning_rate=0.1,0.01,0.001
          batch_size=32,64
```

## Limitations

While powerful, the extended syntax has a few limitations:

1. Parameters with spaces or special characters must be quoted
2. The syntax is limited to parameter values (not parameter names)
3. Extremely large parameter spaces might be inefficient to expand in memory

## Best Practices

1. **Start Small**: Test your syntax with a few values before scaling to large sweeps
2. **Use Clear Separators**: Maintain readability by separating parameter groups
3. **Limit Combinatorial Explosion**: Be careful with multiple parameters that combine to create very large sweep spaces
4. **Validate Your Expansions**: Use the `--dry-run` flag to check expanded commands before executing
5. **Prefer Comma Notation for Log Scales**: For learning rates and other parameters that vary by orders of magnitude, use explicit comma notation (`0.1,0.01,0.001`) instead of ranges