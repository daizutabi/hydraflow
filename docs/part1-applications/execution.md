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

1. Initialize the Hydra environment
2. Create a new MLflow run
3. Load the configuration with default values
4. Execute your function decorated with `@hydraflow.main`
5. Save all outputs and logs

## Command-line Override Syntax

HydraFlow (via Hydra) provides a powerful syntax for overriding configuration
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

By default, HydraFlow organizes outputs in the following directory structure:

```
outputs/
├── YYYY-MM-DD/                 # Date
│   └── HH-MM-SS/              # Time
│       └── .hydra/            # Hydra configuration
└── multirun/                   # Created in multirun mode
    └── YYYY-MM-DD/
        └── HH-MM-SS/
            ├── 0/             # Run 0
            │   └── .hydra/
            ├── 1/             # Run 1
            │   └── .hydra/
            └── ...
```

Each run also creates an entry in the MLflow tracking directory:

```
mlruns/
├── 0/                          # Experiment ID
│   ├── run_id_1/              # Run ID
│   │   ├── artifacts/
│   │   ├── metrics/
│   │   ├── params/
│   │   └── tags/
│   └── run_id_2/
└── meta.yaml
```

## Customizing Output Directories

You can customize the output directory structure in your Hydra configuration:

```yaml
# conf/config.yaml
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
```

## Experiment Name and Tracking

HydraFlow uses the application name as the experiment name in MLflow. You can
customize this:

```python
@hydraflow.main(Config, run_name="custom_experiment")
def train(run, cfg: Config) -> None:
    # Your code here
```

You can also set the MLflow tracking URI:

```bash
# Set before running the application
export MLFLOW_TRACKING_URI=http://localhost:5000

# Or set in code
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
```

## Working with Output Files

HydraFlow automatically changes the working directory to the output directory
when executing your application. This means you can simply use relative paths
for output files:

```python
@hydraflow.main(Config)
def train(run, cfg: Config) -> None:
    # Files will be saved in the run's output directory
    torch.save(model.state_dict(), "model.pt")

    # To get the original working directory
    original_dir = hydra.utils.get_original_cwd()
    data_path = os.path.join(original_dir, "data/train.csv")
```

## Logging Metrics and Artifacts

During execution, you can log metrics and artifacts to MLflow:

```python
@hydraflow.main(Config)
def train(run, cfg: Config) -> None:
    # Train model
    for epoch in range(cfg.epochs):
        # ...training code...

        # Log metrics
        hydraflow.log_metric("accuracy", accuracy, step=epoch)
        hydraflow.log_metric("loss", loss, step=epoch)

    # Log artifacts
    hydraflow.log_artifact("model.pt", "Model checkpoint")
    hydraflow.log_artifact("results.csv", "Evaluation results")
```

## Parallel Execution in Multirun Mode

For large parameter sweeps, you can enable parallel execution:

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

## Debugging and Dry Runs

For debugging purposes, you can:

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

## Error Handling

It's important to handle errors properly in your application to ensure that
MLflow properly tracks failed runs:

```python
@hydraflow.main(Config)
def train(run, cfg: Config) -> None:
    try:
        # Your training code
        model = train_model(cfg)
        accuracy = evaluate_model(model)
        hydraflow.log_metric("accuracy", accuracy)
    except Exception as e:
        # Log the error and re-raise
        hydraflow.log_param("error", str(e))
        raise
```

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

HydraFlow's execution system, built on Hydra, provides powerful tools for
running machine learning experiments with different configurations. The
integration with MLflow ensures that all experiment details are tracked
consistently, making your research reproducible and your workflow more
efficient.