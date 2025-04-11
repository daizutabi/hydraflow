# Basic Application

```bash exec="1" workdir="examples"
rm -rf mlruns outputs multirun __pycache__
```

## Project structure

```console exec="1" source="console" workdir="examples"
$ tree
```
## Hydra application

The following example demonstrates how to use a Hydraflow application.

```python title="example.py" linenums="1"
--8<-- "examples/example.py"
```

### Hydraflow's `main` decorator

[`hydraflow.main`][] starts a new MLflow run that logs the Hydra
configuration. The decorated function must have two arguments: `run` and
`cfg`. The `run` argument is the current MLflow run with type
`mlflow.entities.Run`. The `cfg` argument is the Hydra configuration
with type `omegaconf.DictConfig`. You can annotate the arguments with
`Run` and `Config` to get type checking and autocompletion in your IDE,
although the `cfg` argument is not actually an instance of `Config`
(duck typing is used).

```python
@hydraflow.main(Config)
def app(run: Run, cfg: Config) -> None:
    pass
```

## Run the application

### Single-run

Run the Hydraflow application as a normal Python script.

```console exec="1" source="console" workdir="examples"
$ python example.py
```

Check the MLflow CLI to view the experiment.

```console exec="1" source="console" workdir="examples"
$ mlflow experiments search
```

The experiment name comes from the name of the Hydra job.

Check the current directory structure.

```console exec="1" source="console" workdir="examples"
$ tree -a -L 5 --dirsfirst -I '.trash|tags'
```

- `outputs` directory is created in single run mode by Hydra.
- `mlruns` directory is created by MLflow.
- The contents in the `artifacts` directory is
copied from the `outputs` directory by HydraFlow.

### Multi-run

Run the Hydraflow application in multi-run mode.

```console exec="1" source="console" workdir="examples"
$ python example.py -m width=400,600 height=100,200
```

Check the current directory structure.

```console exec="1" source="console" workdir="examples"
$ tree -a -L 5 --dirsfirst -I '.trash|metrics|params|tags|*.yaml'
```

The four runs are added to the same experiment.

Now you can safely delete the `outputs` and `multirun` directories.

```console exec="1" source="console" workdir="examples"
$ rm -rf outputs multirun
$ tree -L 3 --dirsfirst
```
