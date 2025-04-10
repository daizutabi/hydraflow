# Installation

This guide walks you through installing HydraFlow and its dependencies.

## Requirements

HydraFlow requires:

- Python 3.10 or higher
- pip or conda package manager

## Basic Installation

You can install HydraFlow using pip:

```bash
pip install hydraflow
```

This installs the core framework with minimal dependencies.

## Development Installation

For development or to access the latest features, install directly from
the GitHub repository:

```bash
git clone https://github.com/username/hydraflow.git
cd hydraflow
pip install -e ".[dev]"
```

Installing with the `[dev]` extra includes development tools like pytest,
black, and other utilities necessary for contributing to the project.

## Optional Dependencies

HydraFlow offers additional features through optional dependencies:

```bash
# Install with visualization support
pip install "hydraflow[viz]"

# Install with all optional dependencies
pip install "hydraflow[all]"
```

## Verifying Installation

After installation, verify that HydraFlow is correctly installed:

```bash
python -c "import hydraflow; print(hydraflow.__version__)"
```

This should print the current version of HydraFlow.

## Environment Setup

While not required, we recommend using a virtual environment:

### Using venv

```bash
python -m venv hydraflow-env
source hydraflow-env/bin/activate  # On Windows: hydraflow-env\Scripts\activate
pip install hydraflow
```

### Using conda

```bash
conda create -n hydraflow-env python=3.10
conda activate hydraflow-env
pip install hydraflow
```

## Troubleshooting

If you encounter issues during installation:

1. Ensure your Python version is 3.10 or higher
2. Update pip: `pip install --upgrade pip`
3. If installing from source, ensure you have the necessary build tools
   installed for your platform

For persistent issues, please check the
[GitHub issues](https://github.com/username/hydraflow/issues) or open a
new issue with details about your environment and the error message.

## Next Steps

Now that you have installed HydraFlow, proceed to
[Core Concepts](concepts.md) to understand the framework's fundamental
principles.