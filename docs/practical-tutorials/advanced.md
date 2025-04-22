# Automating Complex Workflows

This tutorial demonstrates how to use HydraFlow's workflow automation capabilities to define, manage, and execute complex experiment workflows.

```bash exec="1" workdir="examples"
rm -rf mlruns outputs multirun __pycache__
```

## Prerequisites

Before you begin this tutorial, you should:

1. Understand basic HydraFlow applications (from the [Creating Your First Application](applications.md) tutorial)
2. Have a basic understanding of YAML configuration files

## Project Structure

First, let's examine our project structure:

```console exec="1" workdir="examples" result="nohighlight"
$ tree --noreport
```

In this tutorial, we'll use:

- `example.py`: Our basic HydraFlow application
- `hydraflow.yaml`: A configuration file to define our experiment workflows
- `submit.py`: A helper script for job submission

## Understanding Job Definitions

The `hydraflow.yaml` file allows you to define reusable experiment workflows:

```yaml title="hydraflow.yaml" linenums="1"
--8<-- "examples/hydraflow.yaml"
```

This configuration file defines three different types of jobs:

1. `job_sequential`: A job that runs sequentially
2. `job_parallel`: A job that runs with parallelization
3. `job_submit`: A job that uses a submit command for custom execution

Each job demonstrates different execution patterns and parameter combinations.

## Using the HydraFlow CLI

HydraFlow provides a command-line interface (CLI) for executing and
managing jobs defined in your `hydraflow.yaml` file.
The primary command is `hydraflow run`, which allows you to execute
any job defined in your configuration.

Basic usage:
```bash
hydraflow run <job_name> [overrides]
```

Where:

- `<job_name>` is the name of a job defined in `hydraflow.yaml`
- `[overrides]` are optional Hydra-style parameter overrides

For more details on the CLI,
see the [Job Configuration](../part2-advanced/job-configuration.md#command-line-interface)
documentation.

## Previewing Execution with Dry Run

Before executing our workflows, we can preview what will happen using the `--dry-run` flag:

```console exec="1" source="console" workdir="examples"
$ hydraflow run job_sequential --dry-run
```

From the dry run output, we can observe:

- 2 jobs will be executed (from the `each` parameter combinations)
- Each job contains 3 sweeps (from the `all` range values)
- Each job includes additional options:
    - `hydra.job.name`: The name of the job defined in hydraflow.yaml
    - `hydra.sweep.dir`: A unique but time-ordered directory for each job created by HydraFlow

Standard Hydra creates directories based on the current date and time, which may cause duplication during parallel execution.
HydraFlow solves this problem by creating unique, time-ordered directories for each job.

## Running Sequential Jobs

Let's examine the sequential job configuration:

```yaml
job_sequential:
  run: python example.py
  sets:
    - each: width=100,300
      all: height=100:300:100
```

This job uses the `each` and `all` parameters to run
multiple configuration combinations in sequence:

```console exec="1" source="console" workdir="examples"
$ hydraflow run job_sequential
```

Results of execution:

- An experiment named `job_sequential` is created
- 2×3=6 jobs are executed sequentially
- A progress bar is displayed to track completion

## Running Parallel Jobs

Now let's look at our parallel job configuration:

```yaml
job_parallel:
  run: python example.py
  add: >-
    hydra/launcher=joblib
    hydra.launcher.n_jobs=3
  sets:
    - each: width=200,400
      all: height=100:300:100
```

This job leverages Hydra's parallel execution features using a joblib launcher via `add` parameter:

```console exec="1" source="console" workdir="examples"
$ hydraflow run job_parallel --dry-run
```

```console exec="1" source="console" workdir="examples"
$ hydraflow run job_parallel
```

Results of execution:

- An experiment named `job_parallel` is created
- The same Python script is used but with a different experiment name
- 2 Python commands are executed sequentially
- Each Python command runs 3 jobs in parallel (using the `hydra/launcher=joblib` configuration)

This demonstrates how HydraFlow makes Hydra's powerful parallel execution features easily accessible.

## Using the Submit Command

For more complex execution patterns, HydraFlow provides the `submit` command. Here's our submit job configuration:

```yaml
job_submit:
  submit: python submit.py example.py
  sets:
    - each: width=250:350:100
      all: height=150,250
```

The `submit` command requires two key components:

1. Your HydraFlow application (`example.py` in this case)
2. A command or script that will receive and process a parameter file

Here's our implementation of the submit handler:

```python title="submit.py" linenums="1"
--8<-- "examples/submit.py"
```

How the `submit` command works:

1. HydraFlow generates all parameter combinations based on your job configuration
2. It writes these combinations to a temporary text file (one combination per line)
3. It runs the command specified in the `submit` field of your `hydraflow.yaml`
4. It **appends the temporary file path as the last argument** to your command

For example, with `submit: python submit.py example.py` in your configuration,
the actual executed command will be something like:
```
python submit.py example.py /tmp/hydraflow_parameters_12345.txt
```

Let's see it in action with a dry run:

```console exec="1" source="console" workdir="examples"
$ hydraflow run job_submit --dry-run
```

And now let's run it:

```console exec="1" source="console" workdir="examples"
$ hydraflow run job_submit
```

Our `submit.py` script implements a simple processor that:

1. Accepts two arguments: the application file (`example.py`) and the parameter file
2. Reads each line from the parameter file
3. Runs the application with each set of parameters sequentially

In real-world scenarios, you could customize this handler to:

- Submit jobs to compute clusters (SLURM, PBS, etc.)
- Implement custom scheduling logic
- Distribute workloads based on resource requirements

## Reviewing Results

With HydraFlow, all important data is stored in MLflow, so we can safely delete the Hydra output directories:

```console exec="1" source="console" workdir="examples"
$ rm -rf multirun
```

Let's check the directory structure:

```console exec="1" workdir="examples" result="nohighlight"
$ tree -L 3 --dirsfirst --noreport
```

After cleanup, we can observe:

- There are three experiments (one for each job type)
- Each experiment contains multiple runs
- A total of 16 runs were executed across all jobs

## Summary

In this tutorial, you've learned how to:

1. Define different types of experiment workflows in a `hydraflow.yaml` file
2. Execute sequential and parallel job runs
3. Use the `submit` command for custom execution patterns
4. Preview jobs with dry runs
5. Manage and organize experiment outputs

These workflow automation capabilities allow you to efficiently manage complex experiment configurations, making your machine learning research more organized and reproducible.

## Next Steps

Now that you've learned about workflow automation, try:

- Defining your own custom workflows
- Exploring more complex parameter sweep combinations
- Learning how to [Analyze Results](analysis.md) from your experiments

For more detailed information, refer to:

- [Part 1: Running Applications](../part1-applications/index.md)
- [Part 2: Automating Workflows](../part2-advanced/index.md)
- [Part 3: Analyzing Results](../part3-analysis/index.md)

- d
    - d
