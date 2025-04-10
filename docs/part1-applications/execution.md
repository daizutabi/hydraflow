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

HydraFlow provides CLI tools to work with multirun mode more efficiently than
using long command lines. These tools automatically track execution history and
make it easier to manage complex parameter sweeps. For details on these advanced
capabilities, see Part 3 of the documentation.

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

**Important**: In HydraFlow, the MLflow run's `artifacts` directory serves
as the single source of truth for your experiment data. The Hydra-generated
output directories are only used as temporary locations to store
configuration files and logs, which are then copied to the MLflow artifacts.
After your application completes, you can safely delete the Hydra output
directories without losing any essential information, as everything important
is preserved in the MLflow run.

Users can always access Hydra-generated configuration files and logs by
navigating to the `run_dir/artifacts/.hydra/` directory within the MLflow
run. This directory contains all the configuration files, overrides, and
other Hydra-related files that were used for that specific run, making it
easy to review and reproduce the experiment settings even after the original
Hydra output directories have been deleted.

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

HydraFlow provides a streamlined way to leverage both Hydra's
configuration management and MLflow's experiment tracking.
The execution system is built on standard Hydra functionality,
while experiment tracking utilizes MLflow's capabilities.
By combining these tools, HydraFlow enables reproducible and
efficient machine learning workflows while maintaining
compatibility with the underlying libraries' documentation and ecosystem.