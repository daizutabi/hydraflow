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

## Verifying Installation

After installation, verify that HydraFlow is correctly installed by running
the CLI command:

```bash
hydraflow --help
```

This should display the help message and available commands, confirming that
HydraFlow is properly installed and accessible from your terminal.

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
[GitHub issues](https://github.com/daizutabi/hydraflow/issues) or open a
new issue with details about your environment and the error message.

## Next Steps

Now that you have installed HydraFlow, proceed to
[Core Concepts](concepts.md) to understand the framework's fundamental
principles.