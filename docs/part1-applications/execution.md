# Executing Applications

Once you've defined your HydraFlow application, you need to execute it to
run experiments. This page covers how to run applications, configure them
via command-line arguments, and perform parameter sweeps.

## Basic Execution

To run a HydraFlow application, simply execute the Python script:

```bash
python train.py
```

This will:

1. Set up an MLflow experiment with the same name as the Hydra job name (using `mlflow.set_experiment`). If the experiment doesn't exist, it will be created automatically
2. Create a new MLflow run or reuse an existing one based on the configuration
3. Save the Hydra configuration as an MLflow artifact
4. Execute your function decorated with `@hydraflow.main`
5. Save only `*.log` files from Hydra's output directory as MLflow artifacts

Note that any other artifacts (models, data files, etc.) must be explicitly saved by your code using MLflow's logging functions. The `chdir` option in the `@hydraflow.main` decorator can help with this by changing the working directory to the run's artifact directory, making file operations more convenient.

## Command-line Override Syntax

Hydra provides a powerful syntax for overriding configuration
values:

```bash
# Override simple parameters
python train.py learning_rate=0.001 batch_size=64

# Override nested parameters
python train.py model.hidden_size=1024 optimizer.learning_rate=0.0001

# Override using configuration groups
python train.py model=cnn optimizer=sgd

# Load specific configurations
python train.py --config-name=my_experiment
```

## Working with Multirun Mode

Hydra's multirun mode allows you to execute your application with multiple
configurations in a single command:

```bash
# Run with different learning rates
python train.py -m optimizer.learning_rate=0.1,0.01,0.001

# Run with different model configurations
python train.py -m model=transformer,cnn,lstm

# Sweep over combinations of parameters
python train.py -m optimizer.learning_rate=0.1,0.01 model=transformer,cnn
```

The above command will run your application 6 times with all combinations
of the specified parameters (2 learning rates × 3 model types).

## Output Organization

By default, Hydra organizes outputs in the following directory structure for HydraFlow applications:

```
outputs/
├── YYYY-MM-DD/              # Date
│   └── HH-MM-SS/            # Time
│       └── .hydra/          # Hydra configuration
└── multirun/                # Created in multirun mode
    └── YYYY-MM-DD/
        └── HH-MM-SS/
            ├── 0/           # Run 0
            │   └── .hydra/
            ├── 1/           # Run 1
            │   └── .hydra/
            └── ...
```

Each run also creates an entry in the MLflow tracking directory:

```
mlruns/
├── experiment_id_1/         # Experiment ID
│   ├── run_id_1/            # Run ID
│   │   ├── artifacts/
│   │   │   └── .hydra/      # Hydra config copied here
│   │   │       ├── config.yaml
│   │   │       └── overrides.yaml
│   │   ├── metrics/
│   │   ├── params/
│   │   └── tags/
│   ├── run_id_2/
│   │   └── artifacts/
│   │       └── .hydra/
│   └── meta.yaml            # Contains Hydra job name
└── meta.yaml
```

**Important**: In HydraFlow, the MLflow run's `artifacts` directory serves as the single source of truth for your experiment data. The Hydra-generated output directories are only used as temporary locations to store configuration files and logs, which are then copied to the MLflow artifacts. After your application completes, you can safely delete the Hydra output directories without losing any essential information, as everything important is preserved in the MLflow run.

Users can always access Hydra-generated configuration files and logs by navigating to the `run_dir/artifacts/.hydra/` directory within the MLflow run. This directory contains all the configuration files, overrides, and other Hydra-related files that were used for that specific run, making it easy to review and reproduce the experiment settings even after the original Hydra output directories have been deleted.

## Customizing Output Directories

HydraFlow applications inherit Hydra's ability to customize output directory structures. This can be done through your Hydra configuration:

```yaml
# conf/config.yaml
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
```

For more customization options, refer to the [Hydra documentation on output directory configuration](https://hydra.cc/docs/configure_hydra/workdir/).

## Experiment Name and Tracking

When using HydraFlow, you can leverage MLflow's experiment naming and tracking capabilities:

```python
@hydraflow.main(Config, run_name="custom_experiment")
def train(run, cfg: Config) -> None:
    # Your code here
```

For remote tracking servers and other MLflow configurations, use MLflow's standard approach:

```bash
# Set before running the application
export MLFLOW_TRACKING_URI=http://localhost:5000

# Or set in code
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
```

For more MLflow tracking options, see the [MLflow tracking documentation](https://www.mlflow.org/docs/latest/tracking.html).

## Working with Output Files

The default Hydra behavior of changing working directories applies to HydraFlow applications as well. If you've enabled the `chdir` option in the `@hydraflow.main` decorator:

```python
@hydraflow.main(Config, chdir=True)
def train(run, cfg: Config) -> None:
    # Files will be saved in the run's output directory
    torch.save(model.state_dict(), "model.pt")

    # To get the original working directory (Hydra utility)
    original_dir = hydra.utils.get_original_cwd()
    data_path = os.path.join(original_dir, "data/train.csv")
```

## Logging Metrics and Artifacts

Standard MLflow logging functions work seamlessly in HydraFlow applications:

```python
@hydraflow.main(Config)
def train(run, cfg: Config) -> None:
    # Train model
    for epoch in range(cfg.epochs):
        # ...training code...

        # Log metrics using MLflow
        mlflow.log_metric("accuracy", accuracy, step=epoch)
        mlflow.log_metric("loss", loss, step=epoch)

    # Log artifacts using MLflow
    mlflow.log_artifact("model.pt")
    mlflow.log_artifact("results.csv")
```

## Parallel Execution in Multirun Mode

Hydra's parallel execution capabilities can be used with HydraFlow applications for efficient parameter sweeps:

```bash
# Run with 4 parallel jobs
python train.py -m optimizer.learning_rate=0.1,0.01,0.001,0.0001 --multirun --hydra.launcher.n_jobs=4
```

This requires setting up a launcher in your Hydra configuration:

```yaml
# conf/config.yaml
hydra:
  launcher:
    _target_: hydra._internal.launcher.basic_launcher.BasicLauncher
    n_jobs: 1  # Default value, override on command line
```

For more launcher options, see the [Hydra documentation on launchers](https://hydra.cc/docs/plugins/joblib_launcher/).

## Debugging and Dry Runs

When debugging HydraFlow applications, you can use Hydra's built-in debugging tools:

1. **Print the resolved configuration**:

```bash
python train.py --cfg=job
```

2. **Run without Hydra output redirection**:

```bash
python train.py hydra.output_subdir=null hydra.job.num=null
```

3. **Show what sweeps will be run without executing**:

```bash
python train.py -m model=transformer,cnn --info
```

These commands use standard Hydra functionality that is available in any HydraFlow application.

## Error Handling

When using MLflow within HydraFlow applications, proper error handling ensures that run status is accurately recorded:

```python
@hydraflow.main(Config)
def train(run, cfg: Config) -> None:
    try:
        # Your training code
        model = train_model(cfg)
        accuracy = evaluate_model(model)
        mlflow.log_metric("accuracy", accuracy)
    except Exception as e:
        # Log the error and re-raise
        mlflow.log_param("error", str(e))
        raise
```

This approach uses standard MLflow APIs to log the error information before allowing the exception to propagate.

## Best Practices

1. **Use Sensible Defaults**: Design your configuration to have sensible defaults
   so that running without arguments produces meaningful results.

2. **Document Command-line Options**: Include a README or help text that
   describes the available command-line options for your application.

3. **Version Control Configurations**: Store configuration files in version
   control to ensure reproducibility.

4. **Organize Complex Sweeps**: For complex parameter sweeps, create separate
   configuration files rather than using long command lines.

5. **Monitor Resource Usage**: Be mindful of resource usage when running large
   parameter sweeps, especially with parallel execution.

## Summary

HydraFlow provides a streamlined way to leverage both Hydra's configuration management and MLflow's experiment tracking. The execution system is built on standard Hydra functionality, while experiment tracking utilizes MLflow's capabilities. By combining these tools, HydraFlow enables reproducible and efficient machine learning workflows while maintaining compatibility with the underlying libraries' documentation and ecosystem.