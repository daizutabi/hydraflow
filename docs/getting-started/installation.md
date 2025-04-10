# Installation

This guide walks you through installing HydraFlow and its dependencies.

## Requirements

HydraFlow requires:

- Python 3.13 or higher
- A package manager (pip or uv)

## Basic Installation

You can install HydraFlow using your preferred package manager:

### Using pip

```bash
pip install hydraflow
```

### Using uv

[uv](https://github.com/astral-sh/uv) is a modern, fast Python package manager:

```bash
uv pip install hydraflow
```

These commands install the core framework with minimal dependencies.

## Development Installation

For development or to access the latest features, install directly from
the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/username/hydraflow.git
cd hydraflow

# Install with pip
pip install -e ".[dev]"

# Or install with uv
uv pip install -e ".[dev]"
```

Installing with the `[dev]` extra includes development tools like pytest,
black, and other utilities necessary for contributing to the project.

## Optional Dependencies

HydraFlow offers additional features through optional dependencies:

```bash
# Install with visualization support using pip
pip install "hydraflow[viz]"

# Install with visualization support using uv
uv pip install "hydraflow[viz]"

# Install with all optional dependencies
pip install "hydraflow[all]"  # or use uv pip
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
pip install hydraflow  # or use uv pip
```

### Using uv

```bash
uv venv hydraflow-env
source hydraflow-env/bin/activate  # On Windows: hydraflow-env\Scripts\activate
uv pip install hydraflow
```

## Troubleshooting

If you encounter issues during installation:

1. Ensure your Python version is 3.13 or higher
2. Update your package manager:
   - For pip: `pip install --upgrade pip`
   - For uv: `uv self update`
3. If installing from source, ensure you have the necessary build tools
   installed for your platform

For persistent issues, please check the
[GitHub issues](https://github.com/username/hydraflow/issues) or open a
new issue with details about your environment and the error message.

## Next Steps

Now that you have installed HydraFlow, proceed to
[Core Concepts](concepts.md) to understand the framework's fundamental
principles.