# Analyzing Experiment Results

This tutorial demonstrates how to use HydraFlow's powerful analysis
capabilities to work with your experiment results.

```bash exec="1" workdir="examples"
rm -rf mlruns outputs multirun __pycache__
```

## Prerequisites

Before you begin this tutorial, you should:

1. Understand the basic structure of a HydraFlow application
   (from the [Basic Application](applications.md) tutorial)
2. Be familiar with the concept of job definitions
   (from the [Automated Workflows](advanced.md) tutorial)

## Project Setup

We'll start by running several experiments that we can analyze.
We'll execute the three jobs defined in the [Automated Workflows](advanced.md) tutorial:

```console exec="1" source="tabbed-left" workdir="examples" result="nohighlight" tabs="Input|Output"
$ hydraflow run job_sequential
$ hydraflow run job_parallel
$ hydraflow run job_submit
```

```bash exec="1" workdir="examples"
rm -rf multirun __pycache__
```

After running these commands, our project structure looks like this:

```console exec="1" workdir="examples" result="nohighlight"
$ tree -L 3 --dirsfirst --noreport
```

The `mlruns` directory contains all our experiment data.
Let's explore how to access and analyze this data using HydraFlow's API.

## Discovering Runs

### Finding Run Directories

HydraFlow provides the [`iter_run_dirs`][hydraflow.iter_run_dirs]
function to discover runs in your MLflow tracking directory:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> from hydraflow import iter_run_dirs
>>> run_dirs = list(iter_run_dirs("mlruns"))
>>> print(len(run_dirs))
>>> for run_dir in run_dirs[:4]:
...     print(run_dir)
```

This function finds all run directories in your MLflow tracking
directory, making it easy to collect runs for analysis.

### Filtering by Experiment Name

You can filter runs by experiment name to focus on specific experiments:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> print(len(list(iter_run_dirs("mlruns", "job_sequential"))))
>>> names = ["job_sequential", "job_parallel"]
>>> print(len(list(iter_run_dirs("mlruns", names))))
>>> print(len(list(iter_run_dirs("mlruns", "job_*"))))
```

As shown above, you can:

- Filter by a single experiment name
- Provide a list of experiment names
- Use pattern matching with wildcards

## Working with Individual Runs

### Loading a Run

The [`Run`][hydraflow.core.run.Run] class represents a single
experiment run in HydraFlow:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> from hydraflow import Run
>>> run_dirs = iter_run_dirs("mlruns")
>>> run_dir = next(run_dirs)  # run_dirs is an iterator
>>> run = Run(run_dir)
>>> print(run)
>>> print(type(run))
```

You can also use the [`load`][hydraflow.core.run.Run.load]
class method, which accepts both string paths and Path objects:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> Run.load(str(run_dir))
>>> print(run)
```

### Accessing Run Information

Each Run instance provides access to run information and configuration:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> print(run.info.run_dir)
>>> print(run.info.run_id)
>>> print(run.info.job_name)  # Hydra job name = MLflow experiment name
```

The configuration is available through the `cfg` attribute:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> print(run.cfg)
```

### Type-Safe Configuration Access

For better IDE integration and type checking, you can
specify the configuration type:

```python exec="1" source="console" session="results" workdir="examples"
from dataclasses import dataclass

@dataclass
class Config:
    width: int = 1024
    height: int = 768
```

```pycon exec="1" source="console" session="results" workdir="examples"
>>> run = Run[Config](run_dir)
>>> print(run)
```

When you use `Run[Config]`, your IDE will recognize `run.cfg` as
having the specified type, enabling autocompletion and type checking.

### Accessing Configuration Values

The `get` method provides a unified interface to access values from a run:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> print(run.get("width"))
>>> print(run.get("height"))
```

## Adding Custom Implementations

### Basic Implementation

You can extend runs with custom implementation classes to
add domain-specific functionality:

```python exec="1" source="console" session="results" workdir="examples"
from pathlib import Path

class Impl:
    root_dir: Path

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def __repr__(self) -> str:
        return f"Impl({self.root_dir.stem!r})"
```

```pycon exec="1" source="console" session="results" workdir="examples"
>>> run = Run[Config, Impl](run_dir, Impl)
>>> print(run)
```

The implementation is lazily initialized when you first access the `impl` attribute:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> print(run.impl)
>>> print(run.impl.root_dir)
```

### Configuration-Aware Implementation

Implementations can also access the run's configuration:

```python exec="1" source="console" session="results" workdir="examples"
from dataclasses import dataclass, field

@dataclass
class Size:
    root_dir: Path = field(repr=False)
    cfg: Config

    @property
    def size(self) -> int:
        return self.cfg.width * self.cfg.height

    def is_large(self) -> bool:
        return self.size > 100000
```

```pycon exec="1" source="console" session="results" workdir="examples"
>>> run = Run[Config, Size].load(run_dir, Size)
>>> print(run)
>>> print(run.impl)
>>> print(run.impl.size)
```

This allows you to define custom analysis methods that use
both the run's artifacts and its configuration.

## Working with Multiple Runs

### Creating a Run Collection

The [`RunCollection`][hydraflow.core.run_collection.RunCollection]
class helps you analyze multiple runs:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> run_dirs = iter_run_dirs("mlruns")
>>> rc = Run[Config, Size].load(run_dirs, Size)
>>> print(rc)
```

The `load` method automatically creates a `RunCollection` when
given multiple run directories.

### Basic Run Collection Operations

You can perform basic operations on a collection:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> print(rc.first())
>>> print(rc.last())
```

### Filtering Runs

The [`filter`][hydraflow.core.collection.Collection.filter] method
lets you select runs based on various criteria:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> print(rc.filter(width=400))
```

You can use lists to filter by multiple values (OR logic):

```pycon exec="1" source="console" session="results" workdir="examples"
>>> print(rc.filter(height=[100, 300]))
```

Tuples create range filters (inclusive):

```pycon exec="1" source="console" session="results" workdir="examples"
>>> print(rc.filter(height=(100, 300)))
```

You can even use custom filter functions:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> print(rc.filter(lambda r: r.impl.is_large()))
```

### Finding Specific Runs

The [`get`][hydraflow.core.collection.Collection.get] method
returns a single run matching your criteria:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> run = rc.get(width=250, height=(100, 200))
>>> print(run)
>>> print(run.impl)
```

### Converting to DataFrames

For data analysis, you can convert runs to a Polars DataFrame:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> print(rc.to_frame("width", "height", "size"))
```

You can add custom columns using callables:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> print(rc.to_frame("width", "height", is_large=lambda r: r.impl.is_large()))
```

Functions can return lists for multiple values:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> def to_list(run: Run) -> list[int]:
...     return [2 * run.get("width"), 3 * run.get("height")]
>>> print(rc.to_frame("width", from_list=to_list))
```

Or dictionaries for multiple named columns:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> def to_dict(run: Run) -> dict[int, str]:
...     width2 = 2 * run.get("width")
...     name = f"h{run.get('height')}"
...     return {"width2": width2, "name": name}
>>> print(rc.to_frame("width", from_dict=to_dict))
```

### Grouping Runs

The [`group_by`][hydraflow.core.collection.Collection.group_by]
method organizes runs by common attributes:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> grouped = rc.group_by("width")
>>> for key, group in grouped.items():
...     print(key, group)
```

You can group by multiple keys:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> grouped = rc.group_by("width", "height")
>>> for key, group in grouped.items():
...     print(key, group)
```

Adding aggregation functions using the
[`agg`][hydraflow.core.group_by.GroupBy.agg]
method transforms the result into a DataFrame:

```pycon exec="1" source="console" session="results" workdir="examples"
>>> grouped = rc.group_by("width")
>>> df = grouped.agg(n=lambda runs: len(runs))
>>> print(df)
```

## Summary

In this tutorial, you've learned how to:

1. Discover experiment runs in your MLflow tracking directory
2. Load and access information from individual runs
3. Add custom implementation classes for domain-specific analysis
4. Filter, group, and analyze collections of runs
5. Convert run data to DataFrames for advanced analysis

These capabilities enable you to efficiently analyze your
experiments and extract valuable insights from your machine
learning workflows.

## Next Steps

Now that you understand HydraFlow's analysis capabilities, you can:

- Dive deeper into the [Run Class](../part3-analysis/run-class.md)
  and [Run Collection](../part3-analysis/run-collection.md) documentation
- Explore advanced analysis techniques in the
  [Analyzing Results](../part3-analysis/index.md) section
- Apply these analysis techniques to your own machine learning experiments
