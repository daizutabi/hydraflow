# Basic HydraFlow Application

This tutorial demonstrates how to create and run a basic HydraFlow application.

```bash exec="1" workdir="examples"
rm -rf mlruns outputs multirun __pycache__
```

## Project Structure

First, let's examine the project structure:

```console exec="1" workdir="examples" result="nohighlight"
$ tree --noreport
```

In this tutorial, we will only use the `example.py` file.

## Creating a HydraFlow Application

Here's an example of a basic HydraFlow application that defines a simple configuration class and integrates Hydra with MLflow:

```python title="example.py" linenums="1"
--8<-- "examples/example.py"
```

### Application Components

The code above consists of the following key elements:

1. **Configuration Class**: Define typed settings using [`dataclass`](https://docs.python.org/3/library/dataclasses.html)
   ```python
   @dataclass
   class Config:
       width: int = 1024
       height: int = 768
   ```

2. **Configuration Registration**: Register the config class with Hydra's `ConfigStore`
   ```python
   cs = ConfigStore.instance()
   cs.store(name="config", node=Config)
   ```

3. **Main Function**: Integrate MLflow and Hydra with the [`hydraflow.main`][hydraflow.main] decorator
   ```python
   @hydraflow.main(Config)
   def app(run: Run, cfg: Config) -> None:
       log.info(run.info.run_id)
       log.info(cfg)
   ```

### Role of the hydraflow.main Decorator

The [`hydraflow.main`][hydraflow.main] decorator serves several purposes:

- Loads and parses Hydra configuration
- Sets up and executes MLflow runs
- Logs Hydra configuration as MLflow artifacts
- Provides type-safe configuration access

The decorated function always takes two arguments:

- `run`: The current MLflow run (type `mlflow.entities.Run`)
- `cfg`: The Hydra configuration (type annotated by your config class)

```python
@hydraflow.main(Config)
def app(run: Run, cfg: Config) -> None:
    # Application logic
```

For more details about the decorator, see the [Main Decorator](../part1-applications/main-decorator.md) documentation.

## Running the Application

### Single-run Mode

Run the HydraFlow application as a normal Python script:

```console exec="1" source="console" workdir="examples"
$ python example.py
```

Use the MLflow CLI to check the created experiment:

```console exec="1" source="console" workdir="examples"
$ mlflow experiments search
```

The experiment name is automatically set from the Hydra job name (in this case, `example`).

Let's examine the current directory structure:

```console exec="1" workdir="examples" result="nohighlight"
$ tree -a -L 5 --dirsfirst -I '.trash|tags' --noreport
```

Key points about the directory structure after execution:

- **`outputs` directory**: Temporary output location created by Hydra in single-run mode
- **`mlruns` directory**: Persistent experiment data storage created by MLflow
- **`artifacts` directory**: Contains configuration files and other artifacts copied from `outputs` by HydraFlow

In HydraFlow, the MLflow `artifacts` directory serves as the single source of truth for experiment data. For more details, see the [Execution](../part1-applications/execution.md#output-organization) documentation.

### Multi-run Mode

Run the application with multiple parameter combinations using Hydra's `-m` (or `--multirun`) flag:

```console exec="1" source="console" workdir="examples"
$ python example.py -m width=400,600 height=100,200
```

This example runs with four combinations:

- width=400, height=100
- width=400, height=200
- width=600, height=100
- width=600, height=200

Check the created directory structure:

```console exec="1" workdir="examples" result="nohighlight"
$ tree -a -L 5 --dirsfirst -I '.trash|metrics|params|tags|*.yaml' --noreport
```

All runs are added to the same MLflow experiment, making it easy to compare results from related parameter sweeps.

With HydraFlow, all important data is stored in MLflow, so you can safely delete the Hydra output directories:

```console exec="1" source="console" workdir="examples"
$ rm -rf outputs multirun
```

The final directory structure should be:

```console exec="1" workdir="examples" result="nohighlight"
$ tree -L 3 --dirsfirst --noreport
```

## Next Steps

Now that you've learned how to create and run a basic application, you can move on to:

- Create reusable job definitions with YAML configuration in [Automated Workflows](advanced.md)
- Analyze experiment results using Run and RunCollection classes in [Results Analysis](analysis.md)

For detailed documentation, refer to:

- [Part 1: Running Applications](../part1-applications/index.md)
- [Part 2: Automating Workflows](../part2-advanced/index.md)
- [Part 3: Analyzing Results](../part3-analysis/index.md)
