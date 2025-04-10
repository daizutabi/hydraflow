# Quickstart

## Hydra application

The following example demonstrates how to use a Hydraflow application.

```python title="apps/quickstart.py" linenums="1"
--8<-- "apps/quickstart.py"
```

### Hydraflow's `main` decorator

[`hydraflow.main`][] starts a new MLflow run that logs the Hydra configuration.
The decorated function must have two arguments: `run` and `cfg`.
The `run` argument is the current MLflow run with type `mlflow.entities.Run`.
The `cfg` argument is the Hydra configuration with type `omegaconf.DictConfig`.
You can annotate the arguments with `Run` and `Config` to get type checking and
autocompletion in your IDE, although the `cfg` argument is not actually an
instance of `Config` (duck typing is used).

```python
@hydraflow.main(Config)
def app(run: Run, cfg: Config) -> None:
    pass
```

## Run the application

```bash exec="on"
rm -rf mlruns outputs multirun
```

### Single-run

Run the Hydraflow application as a normal Python script.

```console exec="1" source="console"
$ python apps/quickstart.py
```

Check the MLflow CLI to view the experiment.

```console exec="1" source="console"
$ mlflow experiments search
```

The experiment name comes from the name of the Hydra job.

### Multi-run

Run the Hydraflow application with multiple configurations.

```console exec="1" source="console"
$ python apps/quickstart.py -m width=400,600 height=100,200,300
```

## Use Hydraflow API

### Iterate over run's directory

The [`hydraflow.iter_run_dirs`][] function iterates over the run directories.
The first argument is the path to the MLflow tracking root directory
(in most cases, this is `"mlruns"`).

```pycon exec="1" source="console" session="quickstart"
>>> import hydraflow
>>> for run_dir in hydraflow.iter_run_dirs("mlruns"):
...     print(run_dir)
```

Optionally, you can specify the experiment name(s) to filter the runs.

```python
>>> hydraflow.iter_run_dirs("mlruns", "quickstart")
>>> hydraflow.iter_run_dirs("mlruns", ["quickstart1", "quickstart2"])
```

### Load a run

[`Run`][hydraflow.core.run.Run] is a class that represents a *Hydraflow* run,
not an MLflow run.
A `Run` instance is created by passing a `pathlib.Path`
instance that points to the run directory to the `Run` constructor.

```pycon exec="1" source="console" session="quickstart"
>>> from hydraflow import Run
>>> run_dirs = hydraflow.iter_run_dirs("mlruns", "quickstart")
>>> run_dir = next(run_dirs)  # run_dirs is an iterator
>>> run = Run(run_dir)
>>> print(run)
>>> print(type(run))
```

You can use `load` class method to load a `Run` instance,
which accepts a `str` as well as `pathlib.Path`.

```pycon exec="1" source="console" session="quickstart"
>>> Run.load(str(run_dir))
>>> print(run)
```

!!! note
    The use case of `Run.load` is to load multiple `Run` instances from run directories as described below.


The `Run` instance has an `info` attribute that contains information about the run.

```pycon exec="1" source="console" session="quickstart"
>>> print(run.info.run_dir)
>>> print(run.info.run_id)
>>> print(run.info.job_name)  # Hydra job name = MLflow experiment name
```

The `Run` instance has a `cfg` attribute that contains the Hydra configuration.

```pycon exec="1" source="console" session="quickstart"
>>> print(run.cfg)
```

### Configuration type of the run

Optionally, you can specify the config type of the run using the `Run[C]` class.

```pycon exec="1" source="console" session="quickstart"
>>> from dataclasses import dataclass
>>> @dataclass
... class Config:
...     width: int = 1024
...     height: int = 768
>>> run = Run[Config](run_dir)
>>> print(run)
>>> # autocompletion occurs below, for example, run.cfg.height
>>> # run.cfg.[TAB]
```

The `Run[C]` class is a generic class that takes a config type `C` as a type parameter. The `run.cfg` attribute is recognized as `C` type in IDEs, which provides autocompletion and type checking.

### Implementation of the run

Optionally, you can specify the implementation of the run.
Use the `Run[C, I]` class to specify the implementation type.
The second argument `impl_factory` is the implementation factory.
which can be a class or a function to generate the implementation.
The `impl_factory` is called with the run's artifacts directory
as the first and only argument.

```pycon exec="1" source="console" session="quickstart"
>>> from pathlib import Path
>>> class Impl:
...     root_dir: Path
...     def __init__(self, root_dir: Path):
...         self.root_dir = root_dir
...     def __repr__(self) -> str:
...         return f"Impl({self.root_dir.stem!r})"
>>> run = Run[Config, Impl](run_dir, Impl)
>>> print(run)
```

The representation of the `Run` instance includes the implementation type
as shown above.

If you specify the implementation type, the `run.impl` attribute
is lazily initialized at the first time of the `run.impl` attribute access.
The `run.impl` attribute is recognized as `I` type in IDEs, which provides autocompletion and type checking.

```pycon exec="1" source="console" session="quickstart"
>>> print(run.impl)
>>> print(run.impl.root_dir)
>>> # autocompletion occurs below, for example, run.impl.root_dir
>>> # run.impl.[TAB]
```

The `impl_factory` can accept two arguments: the run's artifacts directory
and the run's configuration.

```pycon exec="1" source="console" session="quickstart"
>>> from dataclasses import dataclass, field
>>> @dataclass
>>> class ImplConfig:
...     root_dir: Path = field(repr=False)
...     cfg: Config
>>> run = Run[Config, ImplConfig].load(run_dir, ImplConfig)
>>> print(run)
>>> print(run.impl)
```

### Collect runs

You can collect multiple `Run` instances from run directories
as a collection of runs [`RunCollection`][hydraflow.RunCollection].


```pycon exec="1" source="console" session="quickstart"
>>> from hydraflow import RunCollection
>>> run_dirs = hydraflow.iter_run_dirs("mlruns", "quickstart")
>>> rc = RunCollection(run_dirs)
>>> print(rc)
```

### Retrieve a run

The `RunCollection` instance has a `first` and `last` method that
returns the first and last run in the collection.

```pycon exec="1" source="console" session="quickstart"
>>> run = rc.first()
>>> print(type(run))
```

```pycon exec="1" source="console" session="quickstart"
>>> run = rc.last()
>>> cfg = hydraflow.load_config(run)
>>> print(cfg)
```

The `load_config` function loads the Hydra configuration from the run.

```pycon exec="1" source="console" session="quickstart"
>>> cfg = hydraflow.load_config(run)
>>> print(type(cfg))
>>> print(cfg)
```

### Filter runs

The `filter` method filters the runs by the given key-value pairs.

```pycon exec="1" source="console" session="quickstart"
>>> filtered = rc.filter(width=400)
>>> print(filtered)
```

If the value is a list, the run will be included if the value is in the list.

```pycon exec="1" source="console" session="quickstart"
>>> filtered = rc.filter(height=[100, 300])
>>> print(filtered)
```

If the value is a tuple, the run will be included if the value is between the tuple.
The start and end of the tuple are inclusive.

```pycon exec="1" source="console" session="quickstart"
>>> filtered = rc.filter(height=(100, 300))
>>> print(filtered)
```

### Group runs

The `groupby` method groups the runs by the given key.

```pycon exec="1" source="console" session="quickstart"
>>> grouped = rc.groupby("width")
>>> for key, group in grouped.items():
...     print(key, group)
```

The `groupby` method can also take a list of keys.

```pycon exec="1" source="console" session="quickstart"
>>> grouped = rc.groupby(["height"])
>>> for key, group in grouped.items():
...     print(key, group)
```

### Config dataframe

The `data.config` attribute returns a pandas DataFrame
of the Hydra configuration.

```pycon exec="1" source="console" session="quickstart"
>>> print(rc.data.config)
```

```bash exec="on"
rm -rf mlruns outputs multirun
```
