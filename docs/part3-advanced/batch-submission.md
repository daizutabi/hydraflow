# Batch Submission

HydraFlow provides powerful tools for submitting parameter sweeps to compute
clusters and job schedulers. This enables efficient execution of large-scale
experiments on distributed computing resources.

## Overview

The batch submission capabilities allow you to:

1. Submit multiple parameter configurations as a single batch job
2. Leverage HPC clusters and job schedulers like SLURM, PBS, etc.
3. Efficiently manage resources for large parameter sweeps
4. Track and monitor distributed experiment runs

## Basic Submission

To use batch submission, define a job with the `submit` property:

```yaml
# .hydraflow.yaml
jobs:
  train:
    submit: sbatch --partition=gpu --time=4:00:00
    with: python train.py
    steps:
      - batch: >-
          learning_rate=0.1,0.01,0.001
          batch_size=32,64
```

Run this job using the HydraFlow CLI:

```bash
hydraflow run train
```

## How Batch Submission Works

When you run a job with the `submit` property:

1. HydraFlow expands all parameter combinations from the job steps
2. Creates a temporary file containing all parameter combinations
3. Submits a single batch job that processes all combinations
4. The batch scheduler manages resource allocation and execution

This is more efficient than submitting each parameter combination as a
separate job, which would overwhelm the job scheduler.

## Submission Commands

HydraFlow supports various batch submission commands:

```yaml
# SLURM
submit: sbatch --partition=gpu

# PBS
submit: qsub -q gpu

# LSF
submit: bsub -q gpu

# Basic shell (for local execution)
submit: bash
```

## Common Submission Patterns

### SLURM Example

```yaml
jobs:
  train_large:
    submit: >-
      sbatch
      --partition=gpu
      --time=12:00:00
      --gres=gpu:2
      --mem=32G
      --cpus-per-task=4
    with: python train.py
    steps:
      - batch: >-
          model=large,xlarge
          learning_rate=0.1:0.01:0.001
```

### PBS Example

```yaml
jobs:
  train_distributed:
    submit: >-
      qsub
      -l nodes=1:ppn=4:gpus=1
      -l walltime=8:00:00
      -q gpu
    with: python train.py
    steps:
      - batch: dataset=imagenet,cifar100
```

## Passing Additional Arguments

You can pass additional arguments when running a job:

```bash
# Add arguments to the job
hydraflow run train_large --seed=42 --debug
```

These arguments will be passed to each command in the batch.

## Dry Runs

Before submitting jobs to a cluster, it's a good practice to perform a dry run:

```bash
hydraflow run train_large --dry-run
```

This will output:

1. The submission command
2. The contents of the temporary file with all parameter combinations

## Environment Variables

The submission command can use environment variables:

```yaml
jobs:
  train:
    submit: sbatch --partition=${PARTITION} --time=${MAX_TIME}
    with: python train.py
```

Then run with:

```bash
PARTITION=gpu MAX_TIME=4:00:00 hydraflow run train
```

## Resource Management

For large parameter sweeps, be mindful of:

1. **Total Job Count**: A single batch job can contain hundreds or thousands of
   parameter combinations
2. **Memory Requirements**: Ensure your submission requests enough memory
3. **Time Limits**: Set appropriate time limits based on the most demanding
   parameter combination
4. **Storage Needs**: Consider output data size for all runs combined

## Integration with MLflow

HydraFlow automatically sets the MLflow experiment name to the job name:

```python
# This happens automatically before job execution
mlflow.set_experiment(job.name)
```

This ensures that all runs from the batch job are grouped under the same
experiment in MLflow.

## Custom Python Functions

Instead of shell commands, you can submit Python functions:

```yaml
jobs:
  analyze:
    submit: python -m dispatch.worker
    call: analysis.process_results
    steps:
      - batch: dataset=train,test,validation
```

The `call` property specifies the Python function to execute.

## Advanced Configuration

### Array Jobs (SLURM)

For SLURM clusters, you can use array jobs:

```yaml
jobs:
  train_array:
    submit: >-
      sbatch
      --array=0-100%10
      --partition=gpu
    with: python train.py --array-index=$SLURM_ARRAY_TASK_ID
```

### Multi-Node Jobs

For distributed training across nodes:

```yaml
jobs:
  train_distributed:
    submit: >-
      sbatch
      --nodes=2
      --ntasks-per-node=1
      --cpus-per-task=4
      --gres=gpu:4
    with: python -m torch.distributed.launch train.py
```

## Best Practices

1. **Start Small**: Test with a small parameter sweep before scaling up
2. **Monitor Resources**: Keep an eye on CPU, GPU, and memory usage
3. **Use Job Arrays**: For clusters that support job arrays, use them instead
   of submitting many individual jobs
4. **Checkpoint Models**: Save intermediate results to resume interrupted jobs
5. **Log Key Metrics**: Ensure important metrics are logged to MLflow for
   easy comparison
6. **Clean Up**: Remove temporary files once jobs complete

## Troubleshooting

### Common Issues

- **Resource Exceeded**: Request more memory/time or reduce batch size
- **File Not Found**: Ensure paths are absolute or relative to the working directory
- **Permission Issues**: Check file permissions for scripts and data
- **Environment Problems**: Ensure required packages are available in the
  cluster environment

### Debugging

To troubleshoot submission issues:

1. Use `--dry-run` to see what would be submitted
2. Check submission logs for errors
3. Try running a single configuration locally before submitting batch
4. Verify that your cluster environment has all required dependencies