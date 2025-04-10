# HydraFlow: Seamless ML Experiment Management

<div class="grid cards" markdown>

- 🚀 **Streamlined Experimentation**
  Create, run, and track ML experiments with minimal boilerplate
- ⚙️ **Hydra + MLflow Integration**
  Combine powerful configuration management with robust experiment tracking
- 📈 **Rich Analysis Tools**
  Filter, group, and visualize experiment results with intuitive APIs
- ⚡ **Performance Optimized**
  Parallel processing support for handling thousands of experiments efficiently

</div>

## What is HydraFlow?

HydraFlow seamlessly integrates [Hydra](https://hydra.cc/) and
[MLflow](https://mlflow.org/) to create a powerful framework for machine
learning experimentation. It solves common challenges in ML research and
production workflows:

- **Configuration Management**: Type-safe, hierarchical, and dynamic
  configuration
- **Experiment Tracking**: Automatically log parameters, metrics, and artifacts
- **Results Analysis**: Flexible tools to filter, compare, and visualize
  experiment results
- **Reproducibility**: Ensure experiments can be reliably reproduced with
  exact parameters

## Key Features

**At Development Time:**
- Type-safe configuration with IDE autocompletion
- Declarative experiment definition with dataclasses
- Seamless integration with existing ML pipelines

**During Execution:**
- Parameter sweeps with one command
- Automatic configuration logging
- De-duplication of identical experiments

**After Completion:**
- Powerful filtering and grouping of results
- Conversion to DataFrames for analysis
- Configuration-aware implementation loading

## Quick Installation

```bash
pip install hydraflow
```

**Requirements:** Python 3.12+

## Minimal Example

```python
from dataclasses import dataclass
import hydraflow
import mlflow

@dataclass
class Config:
    learning_rate: float = 0.001
    batch_size: int = 32

@hydraflow.main(Config)
def experiment(run, cfg):
    # Your experiment code here
    mlflow.log_metric("accuracy", 0.95)

if __name__ == "__main__":
    experiment()
```

Run this with parameter variations in one command:

```bash
python experiment.py -m learning_rate=0.01,0.001,0.0001 batch_size=16,32,64
```

## Post-experiment Analysis

After running your experiments, analyze the results with HydraFlow's
powerful API:

```python
from hydraflow import Run, RunCollection

# Load all runs from the "experiment" experiment
runs = Run.load(hydraflow.iter_run_dirs("mlruns", "experiment"))

# Filter runs by configuration parameters
best_runs = runs.filter(learning_rate=0.001, batch_size=32)

# Convert to DataFrame for further analysis
df = runs.to_frame("learning_rate", "batch_size",
                   accuracy=lambda run: run.get("metrics.accuracy"))
```

<!--
## Explore HydraFlow

<div class="grid cards" markdown>

- 📖 [**Getting Started**](usage/quickstart.md)
  Learn the basics of HydraFlow with a step-by-step guide
- 🧩 [**API Reference**](api/index.md)
  Detailed documentation of HydraFlow's classes and functions
- 💻 [**CLI Tools**](cli/index.md)
  Discover HydraFlow's command-line utilities
- 💡 [**Advanced Usage**](advanced/index.md)
  Tips, tricks, and best practices for complex workflows

</div>
-->
