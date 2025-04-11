# Updating Run Configurations

As machine learning projects evolve, configuration structures often change.
The [`update`][hydraflow.core.run.Run.update] method in HydraFlow provides a
powerful way to handle these changes and work with runs from different
periods in your project's lifecycle.

## The Configuration Evolution Problem

A common challenge in ML experimentation is dealing with changing configuration
schemas:

1. You start with a specific configuration structure
2. As your project evolves, you add new parameters
3. Now you have a mix of old and new runs with different configuration schemas
4. You want to analyze all runs together, but filtering becomes problematic

For example, imagine you've been training image models with a fixed aspect ratio,
but later decided to parameterize this aspect ratio:

```python
# Old configuration (fixed aspect ratio)
@dataclass
class ModelConfig:
    width: int = 256
    height: int = 256
    # aspect_ratio is implicitly 1:1

# New configuration (parameterized aspect ratio)
@dataclass
class ModelConfig:
    width: int = 256
    height: int = 256
    aspect_ratio: float = 1.0  # New parameter!
```

When you try to filter runs by `aspect_ratio`, older runs will lack this
parameter, making consistent analysis difficult.

## Using the Update Method

The `update` method solves this problem by allowing you to add missing
configuration parameters to runs without altering existing values:

```python
from hydraflow import Run

# Load a mix of old and new runs
runs = Run.load(["old_run_dir", "new_run_dir"])

# Add aspect_ratio to runs that don't have it
for run in runs:
    run.update("aspect_ratio", 1.0)  # Add default value if missing

# Now you can filter by aspect_ratio
square_runs = runs.filter(aspect_ratio=1.0)
```

The `update` method only adds values if the key doesn't already exist. For runs
that already have an `aspect_ratio` parameter, the original value is preserved.

## Batch Updates with RunCollection

To simplify updating multiple runs, you can use the
[`RunCollection.update`][hydraflow.core.run_collection.RunCollection.update] method:

```python
# Update all runs at once
runs.update("aspect_ratio", 1.0)

# Now all runs have the aspect_ratio parameter
```

## Dynamic Updates with Callables

You can provide a callable function instead of a fixed value to compute
parameters dynamically:

```python
# Calculate aspect_ratio from width and height
def calculate_aspect_ratio(run: Run) -> float:
    width = run.get("width", 0)  # Use default value if key doesn't exist
    height = run.get("height", 0)
    if height == 0:
        return 1.0
    return width / height

# Update with calculated values
runs.update("aspect_ratio", calculate_aspect_ratio)
```

## Updating Multiple Parameters

You can update multiple related parameters at once by using a tuple of keys:

```python
# Update both width and height parameters
runs.update(
    ("width", "height"),
    (640, 480)
)

# Update with calculated values
def calculate_dimensions(run: Run) -> tuple[int, int]:
    base_size = run.get("base_size", 256)  # Default value if key doesn't exist
    return (base_size, base_size)

runs.update(("width", "height"), calculate_dimensions)
```

## Forcing Updates

By default, `update` won't modify existing values. To override this behavior,
use the `force` parameter:

```python
# Force update even if the parameter already exists
runs.update("aspect_ratio", 1.0, force=True)
```

## Best Practices

1. **Add Documentation**: Comment your code to explain why updates are needed,
   especially for future reference.

2. **Use Consistent Defaults**: When adding missing parameters, use sensible
   defaults that reflect the implicit values of older runs.

3. **Consider Dynamic Updates**: When possible, compute missing values from
   existing parameters to maintain consistency.

4. **Update Early**: Apply updates early in your analysis pipeline, before
   filtering or grouping.

## Summary

The `update` method enables you to work with runs that have evolving
configuration schemas. By adding missing parameters, you can treat old and
new runs uniformly, enabling consistent analysis across your project's
lifetime. This approach provides a form of "duck typing" for run
configurations, allowing you to analyze runs based on their functional
properties rather than their exact structure.