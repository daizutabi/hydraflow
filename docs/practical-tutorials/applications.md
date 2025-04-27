# Creating Your First HydraFlow Application

This tutorial demonstrates how to create and run a basic HydraFlow
application that integrates Hydra's configuration management
with MLflow's experiment tracking.

```bash exec="1" workdir="examples"
rm -rf mlruns outputs multirun __pycache__
```

## Prerequisites

Before you begin this tutorial, you should:

1. Have HydraFlow installed ([Installation Guide](../getting-started/installation.md))
2. Have a basic understanding of Python and machine learning experiments

## Project Structure

First, let's examine our project structure:

```console exec="1" workdir="examples" result="nohighlight"
$ tree --noreport
```

In this tutorial, we will only use the `example.py` file.

## Creating a Basic Application

Let's create a simple HydraFlow application that defines a
configuration class and tracks experiment parameters:

```python title="example.py" linenums="1"
--8<-- "examples/example.py"
```

### Understanding the Key Components

Let's break down the essential parts of this application:

1. **Configuration Class**: A dataclass that defines the parameters for our experiment
    ```python
    @dataclass
    class Config:
        width: int = 1024
        height: int = 768
    ```

    This class defines the structure and default values for our configuration parameters.
    Using Python's dataclass gives us type safety and clear structure.

2. **Main Function**: The core of our application, decorated with `@hydraflow.main`
    ```python
    @hydraflow.main(Config)
    def app(run: Run, cfg: Config) -> None:
        log.info(run.info.run_id)
        log.info(cfg)
    ```

    This function will be executed with the provided configuration. It takes two key parameters:

    - `run`: An MLflow run object that provides access to the current experiment
    - `cfg`: The configuration object with our parameters

### The Power of the Decorator

The [`hydraflow.main`][hydraflow.main] decorator is where the magic happens:

- It registers your configuration class with Hydra's `ConfigStore`
- It sets up an MLflow experiment
- It starts an MLflow run and passes it to your function
- It stores all Hydra configuration and logs as MLflow artifacts

This single decorator seamlessly connects Hydra's configuration capabilities
with MLflow's experiment tracking.

## Running the Application

Now that we understand the code, let's run our application.

### Single-run Mode

First, let's run it in single-run mode:

```console exec="1" source="console" workdir="examples"
$ python example.py
```

When you run the application, HydraFlow automatically:

1. Creates an MLflow experiment named after your application (in this case, "example")
2. Starts a run with the provided configuration
3. Captures logs and artifacts

Let's use the MLflow CLI to verify that our experiment was created:

```console exec="1" source="console" workdir="examples"
$ mlflow experiments search
```

Now, let's examine the directory structure created by Hydra and MlFlow:

```console exec="1" workdir="examples" result="nohighlight"
$ tree -a -L 5 --dirsfirst -I '.trash|tags' --noreport
```

The directory structure shows:

- **`outputs` directory**: Created by Hydra to store the run's outputs
- **`mlruns` directory**: Created by MLflow to store experiment data
- **`artifacts` directory**: Contains configuration files and logs managed by HydraFlow

### Multi-run Mode (Parameter Sweeps)

One of Hydra's most powerful features is the ability to run parameter sweeps.
Let's try this by overriding our configuration parameters:

```console exec="1" source="console" workdir="examples"
$ python example.py -m width=400,600 height=100,200
```

The `-m` flag (or `--multirun`) tells Hydra to run all combinations of
the specified parameters. In this case, we'll run 4 combinations:

- width=400, height=100
- width=400, height=200
- width=600, height=100
- width=600, height=200

Let's see the updated directory structure:

```console exec="1" workdir="examples" result="nohighlight"
$ tree -a -L 5 --dirsfirst -I '.trash|metrics|params|tags|*.yaml' --noreport
```

Notice that all runs are added to the same MLflow experiment, making it
easy to compare results across parameter combinations.

## Cleanup

With HydraFlow, all important data is stored in MLflow,
so you can safely delete the Hydra output directories:

```console exec="1" source="console" workdir="examples"
$ rm -rf outputs multirun
```

After cleanup, the directory structure is much simpler:

```console exec="1" workdir="examples" result="nohighlight"
$ tree -L 3 --dirsfirst --noreport
```

All experiment data remains safely stored in the MLflow directory.

## Summary

In this tutorial, you've learned how to:

1. Create a simple HydraFlow application using the `@hydraflow.main` decorator
2. Define configuration using Python dataclasses
3. Run experiments with default and overridden parameters
4. Perform parameter sweeps using Hydra's multi-run capabilities

This basic pattern forms the foundation for all HydraFlow applications.
As your machine learning workflows grow in complexity,
you can build upon this foundation to create more sophisticated experiments.

## Next Steps

Now that you've learned how to create and run a basic application, try:

- Creating more complex configurations with nested parameters
- Adding actual machine learning code to your application
- Exploring [Automated Workflows](advanced.md) with HydraFlow
- Learning how to [Analyze Results](analysis.md) from your experiments

For detailed documentation, refer to:

- [Part 1: Running Applications](../part1-applications/index.md)
- [Part 2: Automating Workflows](../part2-advanced/index.md)
- [Part 3: Analyzing Results](../part3-analysis/index.md)
