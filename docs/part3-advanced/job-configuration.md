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
| `batch`  | Parameter sweep that creates separate job executions - runs sequentially by default, or in parallel when using `submit` |
| `args`   | Arguments that apply to each batch execution - can also use extended sweep syntax but all expansions are included in a single command |
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
          learning_rate=0.1,0.01,0.001,0.0001

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

The distinction between `batch` and `args` is crucial for understanding how HydraFlow constructs and executes commands:

```yaml
jobs:
  example:
    run: python app.py
    steps:
      # Example 1: Using batch
      - batch: b=3,4
        args: a=1,2
        # HydraFlow constructs TWO separate commands:
        # 1. python app.py b=3 a=1,2
        # 2. python app.py b=4 a=1,2
```

#### How Command Execution Works

HydraFlow's role is to **construct command-line arguments** based on your parameter definitions and then execute your specified command with those arguments. What happens after that depends entirely on the command you've specified:

##### With `run` property:

```yaml
run: python app.py
```

HydraFlow executes each generated command sequentially using `subprocess.run`:
1. First command runs and completes
2. Then the next command runs
3. And so on...

If your command completes quickly (like submitting a job to a scheduler), the next command will execute almost immediately.

##### With `submit` property:

```yaml
submit: sbatch
```

HydraFlow:
1. Creates a temporary text file containing all generated command lines
2. Executes a single command passing this file: `sbatch temp_file.txt`
3. Your submission command is responsible for processing the file contents

This is optimal for cluster schedulers that can efficiently process multiple jobs from a file.

### Examples with Explanation

```yaml
jobs:
  sequential_processing:
    run: python process.py
    steps:
      - batch: dataset=train,test,validation
        args: model=resnet,vgg
        # Executes sequentially:
        # 1. python process.py dataset=train model=resnet,vgg
        # 2. python process.py dataset=test model=resnet,vgg
        # 3. python process.py dataset=validation model=resnet,vgg

  cluster_submission:
    submit: sbatch --partition=gpu
    with: python train.py
    steps:
      - batch: learning_rate=0.1,0.01,0.001
        # Creates a file with lines:
        # python train.py learning_rate=0.1
        # python train.py learning_rate=0.01
        # python train.py learning_rate=0.001
        # Then executes: sbatch --partition=gpu temp_file.txt
```

HydraFlow focuses on making parameter sweep definition simple and flexible. The actual execution behavior (sequential, parallel, distributed) depends on the commands you specify, giving you complete control over how your workflows run.

## Workflow Integration Examples

HydraFlow is designed to integrate with various cluster environments and
batch systems. Below are examples showing how to configure HydraFlow with different
submission systems:

### Slurm (sbatch)

```yaml
jobs:
  train:
    submit: sbatch --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1
    with: python train.py
    steps:
      - batch: model=resnet,vgg dataset=imagenet,cifar
```

This submits a single job to Slurm that processes all combinations.

### PBS

```yaml
jobs:
  train:
    submit: qsub -l select=1:ncpus=4:ngpus=1 -q gpu_queue
    with: python train.py
    steps:
      - batch: model=resnet,vgg dataset=imagenet,cifar
```

### SGE (Sun Grid Engine)

```yaml
jobs:
  train:
    submit: qsub -l gpu=1 -q ml.q
    with: python train.py
    steps:
      - batch: model=resnet,vgg dataset=imagenet,cifar
```

### Local Parallel Execution

You can use GNU Parallel for local parallelization:

```yaml
jobs:
  train:
    submit: parallel -j 4
    with: python train.py
    steps:
      - batch: model=resnet,vgg dataset=imagenet,cifar
```

### Kubernetes

For Kubernetes integration, you might use a helper script:

```yaml
jobs:
  train:
    submit: ./k8s_submit.sh
    with: python train.py
    steps:
      - batch: model=resnet,vgg dataset=imagenet,cifar
```

Where `k8s_submit.sh` might create a job from each line in the input file.

### Custom Submission Systems

You can create a custom submission script for any environment:

```yaml
jobs:
  train:
    submit: ./my_custom_submitter.py --resource=gpu
    with: python train.py
    steps:
      - batch: model=resnet,vgg dataset=imagenet,cifar
```

The key is that HydraFlow is agnostic to the submission system - it simply
prepares the parameter combinations and passes them to your command. This
flexibility allows HydraFlow to work with virtually any computing environment
or batch system.