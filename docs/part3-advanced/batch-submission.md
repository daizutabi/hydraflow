# Batch Submission

HydraFlow provides efficient batch submission capabilities for running
experiments on clusters and job schedulers. This page explains how to configure
and use batch submission with HydraFlow.

## Basic Batch Submission

To submit jobs to a cluster, use the `submit` command in your job definition:

```yaml
jobs:
  train:
    submit: sbatch --partition=gpu --nodes=1 job.sh
    sets:
      - batch: >-
          model=small,large
          learning_rate=0.1,0.01
```

This configuration will submit the generated commands to the cluster using
the `sbatch` command.

## How Batch Submission Works

When using the `submit` command:

1. HydraFlow generates all parameter combinations based on your sets
2. These combinations are written to a temporary file
3. Your submission command is executed with the temporary file as input
4. The scheduler processes each line as a separate job

This approach is optimal for efficiently submitting many jobs to a cluster.

## Cluster-Specific Examples

### Slurm

Slurm is one of the most common job schedulers. Here's an example of
submitting jobs to a Slurm cluster:

```yaml
jobs:
  train:
    submit: sbatch --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1
    with: python train.py
    sets:
      - batch: model=resnet,vgg dataset=imagenet,cifar
```

This submits a batch file to Slurm that processes all combinations,
allocating the specified resources.

### PBS/Torque

For PBS/Torque clusters:

```yaml
jobs:
  train:
    submit: qsub -l select=1:ncpus=4:ngpus=1 -q gpu_queue
    with: python train.py
    sets:
      - batch: model=resnet,vgg dataset=imagenet,cifar
```

### SGE (Sun Grid Engine)

For SGE clusters:

```yaml
jobs:
  train:
    submit: qsub -l gpu=1 -q ml.q
    with: python train.py
    sets:
      - batch: model=resnet,vgg dataset=imagenet,cifar
```

### LSF

For LSF clusters:

```yaml
jobs:
  train:
    submit: bsub -q gpu -n 4 -gpu "num=1"
    with: python train.py
    sets:
      - batch: model=resnet,vgg dataset=imagenet,cifar
```

## Using Job Templates

Many schedulers support job templates or script files. You can use these
with HydraFlow:

```yaml
jobs:
  train:
    submit: sbatch job_template.sh
    sets:
      - batch: model=resnet,vgg
```

Where `job_template.sh` might look like:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=16G

# This script receives the HydraFlow command as input
eval "$@"
```

## Resource Allocation

You can specify different resources for different parameter combinations:

```yaml
jobs:
  train:
    submit: sbatch
    sets:
      # Small models use less resources
      - batch: model=small
        with: SBATCH="--partition=gpu --gres=gpu:1 --mem=8G"

      # Large models need more resources
      - batch: model=large
        with: SBATCH="--partition=gpu --gres=gpu:2 --mem=32G"
```

## Environment Variables

You can set environment variables for your jobs:

```yaml
jobs:
  train:
    submit: sbatch job.sh
    sets:
      - batch: model=small,large
      - with: >-
          CUDA_VISIBLE_DEVICES=0
          OMP_NUM_THREADS=4
```

## Job Arrays

Some schedulers support job arrays for efficiently submitting many
similar jobs. HydraFlow can work with these as well:

```yaml
jobs:
  train:
    submit: sbatch --array=0-3 job_array.sh
    sets:
      - batch: model=small,large learning_rate=0.1,0.01
```

In your job script, you would access the array index:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Get command from the temporary file based on array index
COMMAND=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
eval "$COMMAND"
```

## Local Parallel Execution

You can use GNU Parallel for local parallelization:

```yaml
jobs:
  train:
    submit: parallel -j 4
    with: python train.py
    sets:
      - batch: model=resnet,vgg dataset=imagenet,cifar
```

## Custom Submission Systems

You can create a custom submission script for any environment:

```yaml
jobs:
  train:
    submit: ./my_custom_submitter.py --resource=gpu
    with: python train.py
    sets:
      - batch: model=resnet,vgg dataset=imagenet,cifar
```

## Handling Job Dependencies

For workflows with dependencies, you can use the scheduler's dependency
features:

```yaml
jobs:
  preprocess:
    submit: sbatch --parsable job.sh
    with: python preprocess.py
    sets:
      - batch: dataset=imagenet,cifar

  train:
    submit: sbatch --dependency=afterok:$PREPROCESS_JOB_ID job.sh
    with: python train.py
    sets:
      - batch: model=resnet,vgg
```

You would need to capture and store the job ID from the first submission.

## Monitoring Submitted Jobs

Most job schedulers provide commands to monitor job status:

```bash
# Slurm
squeue -u $USER

# PBS/Torque
qstat -u $USER

# SGE
qstat -u $USER

# LSF
bjobs
```

## Controlling Resource Usage

To optimize resource usage, consider:

1. **Grouping similar jobs**: Jobs with similar resource requirements can be grouped
2. **Job sizing**: Match job requirements to your cluster's node sizes
3. **Time limits**: Set realistic time limits to help the scheduler

## Handling Failures

For robust batch submission, consider:

1. **Retry logic**: Add retry logic in your job scripts
2. **Output logs**: Ensure outputs and errors are captured
3. **Checkpointing**: Implement checkpointing to resume from failures

## Best Practices

### Validate Before Submission

Always perform a dry run before submitting to the cluster:

```bash
$ hydraflow run train --dry-run
```

### Start Small

Begin with a small subset of your parameter space to ensure everything works:

```yaml
jobs:
  test_run:
    submit: sbatch job.sh
    sets:
      - batch: model=small
      - args: debug=true max_steps=100
```

### Document Job Requirements

Document resource requirements for different job types:

```yaml
# Resource requirements:
# - Small model: 1 GPU, 8GB memory
# - Medium model: 2 GPUs, 16GB memory
# - Large model: 4 GPUs, 32GB memory
jobs:
  train_small:
    submit: sbatch --gres=gpu:1 --mem=8G job.sh
    # ...
```

### Use Set Environment Variables

Use environment variables to configure your job scripts:

```yaml
jobs:
  train:
    submit: sbatch job.sh
    sets:
      - batch: model=small
        with: >-
          NUM_GPUS=1
          MEMORY=8G

      - batch: model=large
        with: >-
          NUM_GPUS=4
          MEMORY=32G
```

Then in your job script:

```bash
#!/bin/bash
#SBATCH --gres=gpu:$NUM_GPUS
#SBATCH --mem=$MEMORY

# Run the command
eval "$@"
```

### Centralize Job Scripts

Maintain job template scripts in a central location:

```
project/
├── hydraflow.yaml
└── cluster/
    ├── gpu_job.sh
    ├── cpu_job.sh
    └── debug_job.sh
```

### Version Control

Keep your job scripts and `hydraflow.yaml` in version control to ensure
reproducibility of your experiments.