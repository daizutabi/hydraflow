# Main Decorator

The [`hydraflow.main`][hydraflow.main] decorator is the central component
for creating HydraFlow applications. It bridges Hydra's configuration
management with MLflow's experiment tracking, automatically setting up
the experiment environment.

## Basic Usage

Here's how to use the main decorator in its simplest form:

```python
from dataclasses import dataclass
from mlflow.entities import Run
import hydraflow

@dataclass
class Config:
    learning_rate: float = 0.01
    batch_size: int = 32

@hydraflow.main(Config)
def train(run: Run, cfg: Config) -> None:
    print(f"Training with learning_rate={cfg.learning_rate}")
    # Your training code here

if __name__ == "__main__":
    train()
```

## Function Signature

The function decorated with [`@hydraflow.main`][hydraflow.main] must accept
two parameters:

1. `run`: The current run object of type `mlflow.entities.Run`, which can be used to access run
   information and log additional metrics or artifacts.

2. `cfg`: The configuration object containing all parameters, populated from
   Hydra's configuration system and command-line overrides.

## Type Annotations

The `cfg` parameter should be annotated with your configuration class for type
checking and IDE auto-completion. This is particularly useful when working
with complex configurations:

```python
@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int

@dataclass
class DataConfig:
    path: str
    validation_split: float

@dataclass
class Config:
    training: TrainingConfig
    data: DataConfig
    seed: int = 42

@hydraflow.main(Config)
def train(run: Run, cfg: Config) -> None:
    # Type-checked access to nested configuration
    lr = cfg.training.learning_rate
    data_path = cfg.data.path

    # Your training code here
```

## Using MLflow APIs

Within a function decorated with [`@hydraflow.main`][hydraflow.main], you have
access to standard MLflow logging functions:

```python
import mlflow

@hydraflow.main(Config)
def train(run: Run, cfg: Config) -> None:
    # Log metrics
    mlflow.log_metric("accuracy", 0.95)

    # Log a set of metrics
    mlflow.log_metrics({
        "precision": 0.92,
        "recall": 0.89,
        "f1_score": 0.90
    })

    # Log artifacts
    mlflow.log_artifact("model.pkl")

    # Log parameters not included in the config
    mlflow.log_param("custom_param", "value")
```

## Run Identification and Reuse

One of HydraFlow's key features is automatic run identification and reuse. By default,
if a run with the same configuration already exists within an experiment, HydraFlow
will reuse that existing run instead of creating a new one.

This behavior is particularly valuable in computation clusters where preemption
(forced termination by the system) can occur. If your job is preempted before
completion, you can simply restart it, and HydraFlow will automatically continue
with the existing run, allowing you to resume from checkpoints.

```python
from pathlib import Path

@hydraflow.main(Config)
def train(run: Run, cfg: Config) -> None:
    # If this exact configuration was run before but interrupted,
    # the same Run object will be reused
    checkpoint_path = Path("checkpoint.pt")

    if checkpoint_path.exists():
        print(f"Resuming from checkpoint in run: {run.info.run_id}")
        # Load checkpoint and continue training
    else:
        print(f"Starting new training in run: {run.info.run_id}")
        # Start training from scratch
```

This default behavior improves efficiency by:

- Avoiding duplicate experiments with identical configurations
- Enabling graceful recovery from system interruptions
- Reducing wasted computation when jobs are preempted
- Supporting iterative development with checkpointing

## Automatic Skipping of Completed Runs

HydraFlow automatically skips runs that have already completed successfully. This is especially valuable in environments where jobs are automatically restarted after preemption. Without requiring any additional configuration, HydraFlow will:

1. Identify already completed runs with the same configuration
2. Skip re-execution of those runs
3. Proceed only with runs that were interrupted or not yet executed

```python
@hydraflow.main(Config)
def train(run: Run, cfg: Config) -> None:
    # If this configuration was already successfully run before,
    # the function won't even be called - HydraFlow automatically
    # skips it and returns immediately

    print(f"This run is either new or was previously interrupted: {run.info.run_id}")
    # Your training code here
```

This automatic skipping behavior:

- Prevents redundant computation in multi-job or batch scenarios
- Handles preemption recovery efficiently in cluster environments
- Reduces resource usage by avoiding unnecessary re-execution
- Works seamlessly without requiring explicit handling in your code

## Advanced Features

The `hydraflow.main` decorator supports several keyword arguments that enhance its functionality. All these options are set to `False` by default and must be explicitly enabled when needed:

### Working Directory Management (`chdir`)

Control whether the working directory changes to the run's artifact directory:

Change the current working directory to the run's artifact directory during execution:

```python
@hydraflow.main(Config, chdir=True)
def train(run: Run, cfg: Config) -> None:
    # Working directory is now the run's artifact directory
    # Useful for relative path references
    with open("results.txt", "w") as f:
        f.write("Results will be saved as an artifact in the run")
```

This option is beneficial when:

- You need to save or access files using relative paths
- Your code relies on local file operations within the experiment directory
- You want artifacts to be automatically associated with the current run
- You're working with libraries that expect files in the current directory

### Forcing New Runs (`force_new_run`)

Override the default run identification and reuse behavior by always creating a new run, even when identical configurations exist:

```python
@hydraflow.main(Config, force_new_run=True)
def train(run: Run, cfg: Config) -> None:
    # This will always create a new run, even if identical
    # configurations exist in the experiment
    print(f"Fresh run created: {run.info.run_id}")
```

This option is useful when:

- You want to test the reproducibility of your experiments
- You need to compare results across multiple identical runs
- You've made changes to external dependencies not captured in the configuration
- You want to avoid the run identification mechanism for debugging purposes

### Rerunning Finished Experiments (`rerun_finished`)

Override the automatic skipping of completed runs by explicitly allowing rerunning of experiments that have already finished:

```python
@hydraflow.main(Config, rerun_finished=True)
def train(run: Run, cfg: Config) -> None:
    # Runs that have FINISHED status will be rerun instead of skipped
    # The same run ID will be reused
    print(f"Run may be rerunning even if it completed successfully: {run.info.run_id}")
```

This option is valuable when:

- You need to regenerate artifacts or metrics from a successful run
- You've improved your logging or analysis and want to apply it to previous runs
- You're iteratively refining experiments without changing their configuration
- You suspect that a "successful" run may have had undetected issues

### Matching Based on Overrides (`match_overrides`)

Match runs based on command-line overrides instead of the full configuration:

```python
@hydraflow.main(Config, match_overrides=True)
def train(run: Run, cfg: Config) -> None:
    # Runs will be matched based on CLI overrides
    # rather than the complete configuration contents
    print(f"Run ID: {run.info.run_id}")
```

This option is particularly useful when:

- You have large default configurations but only care about specific parameters
- You want to group runs by the parameters that were explicitly overridden
- You're iterating on experiments with command-line variations
- Your configuration contains volatile or automatically generated values

## Best Practices

1. **Keep Configuration Classes Focused**: Break down complex configurations
   into logical components using nested dataclasses.

2. **Use Type Annotations**: Always annotate your function parameters for
   better IDE support and type checking.

3. **Log Important Information**: Log all relevant metrics, parameters, and
   artifacts to ensure reproducibility.

4. **Handle Errors Gracefully**: Implement proper error handling inside your
   main function to avoid losing experiment data.

## Summary

The [`hydraflow.main`][hydraflow.main] decorator simplifies the integration of
Hydra and MLflow, handling configuration management and experiment tracking
automatically. This allows you to focus on your experiment implementation
while ensuring that all relevant information is properly tracked and organized.