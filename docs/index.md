# HydraFlow: Streamline ML Experiment Workflows

<div class="grid cards" markdown>

- 🚀 **Define and Run Experiments**
  Combine Hydra's configuration management with MLflow's experiment
  tracking for streamlined experiment workflows
- 📊 **Collect and Analyze Results**
  Gather, filter, and analyze experiment results with type-safe APIs
  for comprehensive insights
- 🔄 **Automate Workflows**
  Define reusable experiment workflows with YAML configuration and
  leverage extended sweep syntax for parameter exploration

</div>

## What is HydraFlow?

HydraFlow seamlessly integrates [Hydra](https://hydra.cc/) and
[MLflow](https://mlflow.org/) to create a comprehensive machine learning
experiment management framework. It provides a complete workflow from defining
experiments to execution and analysis, streamlining machine learning projects
from research to production.

### Key Integration Features

- **Automatic Configuration Tracking**: Hydra configurations are automatically
  saved as MLflow artifacts, ensuring complete reproducibility of experiments
- **Type-safe Configuration**: Leverage Python dataclasses for type-safe
  experiment configuration with full IDE support
- **Unified Workflow**: Connect configuration management and experiment tracking
  in a single, coherent workflow
- **Powerful Analysis Tools**: Analyze and compare experiments using
  configuration parameters captured from Hydra

### Hydra + MLflow = More Than the Sum of Parts

HydraFlow goes beyond simply using Hydra and MLflow side by side:

- **Parameter Sweep Integration**: Run Hydra multi-run sweeps with automatic
  MLflow experiment organization
- **Configuration-Aware Analysis**: Filter and group experiment results using
  Hydra configuration parameters
- **Reproducible Experiments**: Ensure experiments can be reliably reproduced
  with configuration-based definitions
- **Implementation Support**: Extend experiment analysis with custom
  domain-specific implementations

## Quick Installation

```bash
pip install hydraflow
```

**Requirements:** Python 3.13+

## Documentation Structure

The HydraFlow documentation is organized as follows:

<div class="grid cards" markdown>

- :material-book-open-variant: [**Getting Started**](getting-started/index.md)
  Install HydraFlow and learn core concepts
- :material-school: [**Practical Tutorials**](practical-tutorials/index.md)
  Learn through hands-on examples and real use cases
- :material-rocket-launch: [**Part 1: Running Applications**](part1-applications/index.md)
  Define and execute HydraFlow applications
- :material-cogs: [**Part 2: Automating Workflows**](part2-advanced/index.md)
  Build advanced experiment workflows
- :material-magnify: [**Part 3: Analyzing Results**](part3-analysis/index.md)
  Collect and analyze experiment results
- :material-code-tags: [**API Reference**](api/hydraflow/README.md)
  Detailed documentation of classes and methods

</div>

## Getting Started

Begin your journey with HydraFlow through our introductory guides:

<div class="grid cards" markdown>

- :material-book-open-variant: [**Installation Guide**](getting-started/installation.md)
  Install and set up HydraFlow
- :material-school: [**Core Concepts**](getting-started/concepts.md)
  Learn the key concepts and design principles of HydraFlow
- :material-file-code: [**Practical Tutorials**](practical-tutorials/index.md)
  Hands-on examples to understand HydraFlow in practice

</div>