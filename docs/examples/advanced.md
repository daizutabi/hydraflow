# Automating Workflows

```bash exec="1" workdir="examples"
rm -rf mlruns outputs multirun __pycache__
```

## Project Structure

First, let's examine the project structure:

```console exec="1" source="console" workdir="examples"
$ tree
```

## Job definitions

The `hydraflow.yaml` file contains the job definitions.

```yaml title="hydraflow.yaml" linenums="1"
--8<-- "examples/hydraflow.yaml"
```

## Dry-run

```console exec="1" source="console" workdir="examples"
$ hydraflow run job_sequential --dry-run
```

- you can see 2 jobs will be executed (`each`).
- each job have 3 sweeps (`all`)
- each job contains additional options:
    - `hydra.job.name` : the name of the job from hydraflow.yaml
    -  `hydra.sweep.dir`: unique directory for each job created by hydraflow

Originally Hydra creates a directory based on the current date and time.
But they may duplicate in parallel execution.
HydraFlow creates a unique directory for each job to avoid this problem.

## Run the jobs in sequence

this job uses  `each`

```console exec="1" source="console" workdir="examples"
$ hydraflow run job_sequential
```

- Experiment `job_sequential` created.
- 2x3 jobs executed sequentially.
- progress bar is shown for convinience.

## Run the jobs in parallel

this job uses `each` and `all`

```console exec="1" source="console" workdir="examples"
$ hydraflow run job_parallel --dry-run
```

```console exec="1" source="console" workdir="examples"
$ hydraflow run job_parallel
```
- Experiment `job_parallel` created.
- same python script used but diffrenent experiment name
- 2 python command executed sequentially.
- each python command contains 3 jobs executed in parallel.

## Use submit command

```python title="submit.py" linenums="1"
--8<-- "examples/submit.py"
```

- `submit.py` just runs the command with options for demonstration.
- actual `submit.py` may include submit command to the cluster.

```console exec="1" source="console" workdir="examples"
$ hydraflow run job_submit --dry-run
```

- first line: the commaned to be executed including temporary file.
- second line and following: parameter sets text in the temporary file.

```console exec="1" source="console" workdir="examples"
$ hydraflow run job_submit
```

## Clean up

With HydraFlow, all important data is stored in MLflow, so you can safely delete the Hydra output directories:

```console exec="1" source="console" workdir="examples"
$ rm -rf multirun
$ tree -L 3 --dirsfirst
```

- there are three experiments
- each experiment have 4 or 6 runs
- total 16 runs executed