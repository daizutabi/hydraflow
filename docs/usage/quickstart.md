# Quickstart

## Hydra application

The following example demonstrates how to use a Hydraflow application.

```python title="apps/quickstart.py" linenums="1"
--8<-- "apps/quickstart.py"
```

### Hydraflow's `main` decorator

[`hydraflow.main`][] starts a new MLflow run that logs the Hydra configuration.
The decorated function must have two arguments: `Run` and `Config`.

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

### Multi-run

```console exec="1" source="console"
$ python apps/quickstart.py -m width=400,600 height=100,200,300
```

## Use Hydraflow API

### Run collection

```pycon exec="1" source="console" session="quickstart"
>>> import hydraflow
>>> rc = hydraflow.list_runs("quickstart")
>>> print(rc)
```

### Retrieve a run

```pycon exec="1" source="console" session="quickstart"
>>> run = rc.first()
>>> print(type(run))
```

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

```pycon exec="1" source="console" session="quickstart"
>>> filtered = rc.filter(width=400)
>>> print(filtered)
```

```pycon exec="1" source="console" session="quickstart"
>>> filtered = rc.filter(height=[100, 300])
>>> print(filtered)
```

```pycon exec="1" source="console" session="quickstart"
>>> filtered = rc.filter(height=(100, 300))
>>> print(filtered)
```

### Group runs

```pycon exec="1" source="console" session="quickstart"
>>> grouped = rc.groupby("width")
>>> for key, group in grouped.items():
...     print(key, group)
```

```pycon exec="1" source="console" session="quickstart"
>>> grouped = rc.groupby(["height"])
>>> for key, group in grouped.items():
...     print(key, group)
```

### Config dataframe

```pycon exec="1" source="console" session="quickstart"
>>> print(rc.data.config)
```

```bash exec="on"
rm -rf mlruns outputs multirun
```
