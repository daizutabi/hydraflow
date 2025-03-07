# Quickstart

## Hydra application

The following example demonstrates how to use a Hydraflow application.

```python title="apps/quickstart.py" linenums="1"
--8<-- "apps/quickstart.py"
```

### Hydraflow's `main` decorator

[`hydraflow.main`][] starts a new MLflow run that logs the Hydra configuration.
The decorated function must have two arguments: `Run` and `Config`.
The `Run` argument is the current MLflow run.
The `Config` argument is the Hydra configuration.

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

The experiment name is the name of the Hydra job.

### Multi-run

Run the Hydraflow application with multiple configurations.

```console exec="1" source="console"
$ python apps/quickstart.py -m width=400,600 height=100,200,300
```

## Use Hydraflow API

### Run collection

The `RunCollection` object is a collection of runs.

```pycon exec="1" source="console" session="quickstart"
>>> import hydraflow
>>> rc = hydraflow.list_runs("quickstart")
>>> print(rc)
```

### Retrieve a run

The `RunCollection` object has a `first` and `last` method that
returns the first and last run in the collection.

```pycon exec="1" source="console" session="quickstart"
>>> run = rc.first()
>>> print(type(run))
```

The `load_config` function loads the Hydra configuration from the run.

```pycon exec="1" source="console" session="quickstart"
>>> cfg = hydraflow.load_config(run)
>>> print(type(cfg))
>>> print(cfg)
```

```pycon exec="1" source="console" session="quickstart"
>>> run = rc.last()
>>> cfg = hydraflow.load_config(run)
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
