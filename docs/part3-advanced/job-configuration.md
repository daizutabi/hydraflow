# Job Configuration

HydraFlow provides a powerful job configuration system inspired by GitHub
Actions workflows. This allows you to define complex, multi-step experiment
workflows in a structured YAML format.

## Configuration File

HydraFlow uses a `hydraflow.yaml` configuration file in your project's root
directory. This file defines all of your experiment jobs.

Basic structure:

```yaml
# hydraflow.yaml
jobs:
  job_name:
    run: command_to_run
    steps:
      - batch: parameter_sweep_definition

  another_job:
    call: python.function.to_call
    with: global_options
```

## Defining Jobs

Each job is defined under the `jobs` key and has a unique name:

```yaml
jobs:
  train:
    run: python train.py

  evaluate:
    run: python evaluate.py
```

## Job Properties

Jobs support several key properties:

| Property | Description |
|----------|-------------|
| `run`    | Shell command to execute (e.g., `python script.py`) |
| `call`   | Python function to call (e.g., `module.function`) |
| `submit` | Command for batch submission (e.g., `sbatch`) |
| `with`   | Global options for all steps |
| `steps`  | List of execution steps |
| `name`   | Optional job display name (defaults to key) |

## Basic Examples

### Simple Command Execution

```yaml
jobs:
  train:
    run: python train.py
```

### Executing a Python Function

```yaml
jobs:
  analyze:
    call: analysis.process_results
```

### Submitting to a Cluster

```yaml
jobs:
  batch_train:
    submit: sbatch --partition=gpu
    with: python train.py
```

## Multi-Step Jobs

Jobs can contain multiple steps, each with its own parameter sweep:

```yaml
jobs:
  full_pipeline:
    run: python
    steps:
      - batch: >-
          train.py learning_rate=0.1,0.01,0.001

      - batch: >-
          evaluate.py model=best metrics=accuracy,f1

      - batch: >-
          deploy.py
```

## Step Properties

Each step supports the following properties:

| Property | Description |
|----------|-------------|
| `batch`  | Parameter sweep that creates separate job executions, each running as individual processes (useful for parallel execution on clusters) |
| `args`   | Fixed arguments added to each batch execution, can contain parameter sweeps that will be passed as-is to a single command |
| `with`   | Step-specific options (overrides job-level) |

## Parameter Inheritance

Options defined at the job level cascade down to steps:

```yaml
jobs:
  train:
    run: python train.py
    with: --config=base.yaml
    steps:
      - batch: learning_rate=0.1
        # Inherits --config=base.yaml

      - batch: learning_rate=0.01
        with: --config=modified.yaml
        # Uses modified.yaml instead of base.yaml
```

## Command-Line Interface

Run jobs using HydraFlow's CLI:

```bash
# Run a job
hydraflow run job_name

# Run with additional arguments
hydraflow run job_name --batch_size=64

# Dry run (print commands without executing)
hydraflow run job_name --dry-run
```

## Inspecting Jobs

View job definitions:

```bash
# Show all configured jobs
hydraflow show

# Show details of a specific job
hydraflow show job_name
```

## Advanced Configuration

### Job Dependencies

You can create workflows where jobs depend on each other by running them
in sequence:

```yaml
jobs:
  preprocess:
    run: python preprocess.py

  train:
    run: python train.py
    # Manually run after preprocess completes
```

### Reusing Configuration

Use YAML anchors and aliases for reusable configuration:

```yaml
jobs:
  default_settings: &defaults
    with: >-
      --config=base.yaml
      --seed=42

  train:
    <<: *defaults
    run: python train.py

  evaluate:
    <<: *defaults
    run: python evaluate.py
```

### Environment Variables

Jobs can use environment variables:

```yaml
jobs:
  train:
    run: python train.py
    with: >-
      --data_dir=${DATA_DIR}
      --output_dir=${OUTPUT_DIR}
```

## Best Practices

1. **Organize Jobs Logically**: Group related jobs and give them meaningful names

2. **Minimize Duplication**: Use YAML anchors and aliases to reuse configuration

3. **Version Control**: Keep your `hydraflow.yaml` in version control for reproducibility

4. **Parameterize Paths**: Use environment variables or configuration for paths to data and outputs

5. **Start Simple**: Begin with simple jobs and gradually add complexity

6. **Use Dry Runs**: Validate complex job configurations with `--dry-run` before execution

## Example Configuration

Complete example of a `hydraflow.yaml` file:

```yaml
jobs:
  preprocess:
    run: python preprocess.py
    with: --dataset=imagenet

  train:
    run: python train.py
    with: >-
      --config=configs/base.yaml
      --seed=42
    steps:
      - batch: >-
          model=resnet50,efficientnet
          learning_rate=0.1:0.001:0.0001

      - batch: >-
          model=transformer
          learning_rate=0.01,0.001
          attention_heads=4,8

  evaluate:
    run: python evaluate.py
    with: --metrics=accuracy,precision,recall
    steps:
      - batch: model=best,checkpoint

  deploy:
    run: python deploy.py
    with: --production
```

### Understanding Batch vs Args

The distinction between `batch` and `args` is crucial:

```yaml
jobs:
  example:
    run: python app.py
    steps:
      # Example 1: Using batch
      - batch: model=cnn,transformer
        # Creates TWO separate processes:
        # 1. python app.py model=cnn
        # 2. python app.py model=transformer

      # Example 2: Using args
      - args: model=cnn,transformer
        # Creates ONE process:
        # python app.py model=cnn,transformer

      # Example 3: Combining batch and args
      - batch: model=cnn,transformer
        args: epochs=10,20
        # Creates TWO separate processes:
        # 1. python app.py model=cnn epochs=10,20
        # 2. python app.py model=transformer epochs=10,20
```

This distinction enables:

1. **Parallel Execution**: Each `batch` item can be executed on a different compute node
2. **Parameter Grouping**: Related parameters can be kept together using `args`
3. **Efficient Resource Use**: Critical parameters that require separate resources use `batch`,
   while others can be combined with `args`