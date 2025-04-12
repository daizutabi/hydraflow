# Automating Workflows

This tutorial demonstrates how to automate workflows using HydraFlow's `hydraflow.yaml` configuration.

```bash exec="1" workdir="examples"
rm -rf mlruns outputs multirun __pycache__
```

## Project Structure

First, let's examine the project structure:

```console exec="1" source="console" workdir="examples"
$ tree
```

## Job Definition File

The `hydraflow.yaml` file contains job definitions that serve as the core of HydraFlow's workflow automation capabilities:

```yaml title="hydraflow.yaml" linenums="1"
--8<-- "examples/hydraflow.yaml"
```

This configuration file defines three different jobs:

1. `job_sequential`: A job that runs sequentially
2. `job_parallel`: A job that runs with parallelization
3. `job_submit`: A job that uses a submit command

Each job demonstrates different execution methods and parameter combinations. For detailed information about job configuration, see the [Job Configuration documentation](../part2-advanced/job-configuration.md).

## Dry Run

A dry run allows you to preview what will happen without actually executing commands:

```console exec="1" source="console" workdir="examples"
$ hydraflow run job_sequential --dry-run
```

From the dry run output, we can observe:

- 2 jobs will be executed (from the `each` parameter combinations)
- Each job contains 3 sweeps (from the `all` range values)
- Each job includes additional options:
    - `hydra.job.name`: The name of the job defined in
       `hydraflow.yaml`
    - `hydra.sweep.dir`: A unique directory for each job created by HydraFlow

Standard Hydra creates directories based on the current date and time, which may cause duplication during parallel execution.
HydraFlow solves this problem by creating a unique directory for each job.

## Running Jobs Sequentially

This job uses the `each` and `add` parameters
to run multiple configuration combinations in sequence:

```console exec="1" source="console" workdir="examples"
$ hydraflow run job_sequential
```

Results of execution:

- An experiment named `job_sequential` is created
- 2×3=6 jobs are executed sequentially
- A progress bar is displayed to track completion

## Running Jobs in Parallel

This job uses the `add` parameter
to leverage Hydra's parallel execution features:

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

This demonstrates how HydraFlow makes Hydra's powerful features easily accessible. For more details about launchers, see the [Hydra documentation](https://hydra.cc/docs/plugins/joblib_launcher/).

## Using the Submit Command

The `submit` command requires two key components to work:

1. Your HydraFlow application (`example.py` in this case)
2. A handler script or command that processes the parameter file (`submit.py` in this example)

This pattern separates the experiment definition from the execution strategy:

```python title="submit.py" linenums="1"
--8<-- "examples/submit.py"
```

This handler script (`submit.py`):

- Receives two arguments: the application script path and the parameter file path
- Reads each line from the parameter file
- Executes the application with each set of parameters separately

When you run a job with the `submit` command, HydraFlow:

1. Creates a temporary file containing all parameter combinations
2. Passes both your application file and this parameter file to your handler
3. Lets your handler decide how to distribute or execute the jobs

This separation enables powerful deployment scenarios:
- Submit jobs to compute clusters (SLURM, PBS, etc.)
- Implement custom scheduling logic
- Distribute workloads based on resource availability

Let's see it in action with a dry run:

```console exec="1" source="console" workdir="examples"
$ hydraflow run job_submit --dry-run
```

The dry run output shows:
- The handler command that will be executed with paths to both your application and the parameter file
- The parameter combinations that will be written to the parameter file

Now let's run it:

```console exec="1" source="console" workdir="examples"
$ hydraflow run job_submit
```

For more details about the `submit` command, see the [Job Configuration documentation](../part2-advanced/job-configuration.md#submit).

## Cleanup

With HydraFlow, all important data is stored in MLflow, so you can safely delete the Hydra output directories:

```console exec="1" source="console" workdir="examples"
$ rm -rf multirun
$ tree -L 3 --dirsfirst
```

After cleanup, you can observe:

- There are three experiments
- Each experiment contains 4-6 runs
- A total of 16 runs were executed

Using HydraFlow automates the storage and organization of experiment data, maintaining a reproducible experiment environment.

## Next Steps

After learning about workflow automation, try these next steps:

- Analyze experiment results using the [Run](../part3-analysis/run-class.md) and [RunCollection](../part3-analysis/run-collection.md) classes
- For more complex parameter sweep definitions, refer to the [Sweep Syntax](../part2-advanced/sweep-syntax.md) documentation

For detailed documentation, see:

- [Part 1: Running Applications](../part1-applications/index.md)
- [Part 2: Automating Workflows](../part2-advanced/index.md)
- [Part 3: Analyzing Results](../part3-analysis/index.md)