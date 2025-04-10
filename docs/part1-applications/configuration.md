# Configuration Management

HydraFlow uses a powerful configuration management system based on Python's
dataclasses and Hydra's composition capabilities. This approach provides
type safety, IDE auto-completion, and flexible parameter specification.

## Basic Configuration

The simplest way to define a configuration for a HydraFlow application is
using a Python dataclass:

```python
from dataclasses import dataclass
import hydraflow

@dataclass
class Config:
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 10
    model_name: str = "transformer"

@hydraflow.main(Config)
def train(run, cfg: Config) -> None:
    # Access configuration parameters
    print(f"Training {cfg.model_name} for {cfg.epochs} epochs")
    print(f"Learning rate: {cfg.learning_rate}, Batch size: {cfg.batch_size}")
```

## Type Hints

Adding type hints to your configuration class provides several benefits:

1. **Static Type Checking**: Tools like mypy can catch configuration errors
   before runtime.
2. **IDE Auto-completion**: Your IDE can provide suggestions as you work with
   configuration objects.
3. **Documentation**: Type hints serve as implicit documentation for your
   configuration parameters.

## Nested Configurations

For more complex applications, you can use nested dataclasses to organize
related parameters:

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str = "transformer"
    hidden_size: int = 512
    num_layers: int = 6
    dropout: float = 0.1

@dataclass
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0

@dataclass
class DataConfig:
    batch_size: int = 32
    num_workers: int = 4
    train_path: str = "data/train"
    val_path: str = "data/val"

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    data: DataConfig = DataConfig()
    seed: int = 42
    max_epochs: int = 10

@hydraflow.main(Config)
def train(run, cfg: Config) -> None:
    # Access nested configuration
    model_name = cfg.model.name
    lr = cfg.optimizer.learning_rate
    batch_size = cfg.data.batch_size
```

## Hydra Configuration Files

While dataclasses provide the configuration structure, you can use YAML files
to define default values. Create a `conf` directory in your project with the
following structure:

```
conf/
├── config.yaml        # Main configuration
├── model/             # Model configurations
│   ├── transformer.yaml
│   └── cnn.yaml
├── optimizer/         # Optimizer configurations
│   ├── adam.yaml
│   └── sgd.yaml
└── data/              # Dataset configurations
    ├── mnist.yaml
    └── cifar10.yaml
```

Example `config.yaml`:

```yaml
# @package _global_
defaults:
  - model: transformer
  - optimizer: adam
  - data: mnist
  - _self_

seed: 42
max_epochs: 10
```

Example `model/transformer.yaml`:

```yaml
# @package _group_
name: transformer
hidden_size: 512
num_layers: 6
dropout: 0.1
```

## Command-line Overrides

A key feature of HydraFlow is the ability to override configuration values
from the command line:

```bash
# Override single values
python train.py optimizer.learning_rate=0.0001 max_epochs=20

# Select configuration groups
python train.py model=cnn optimizer=sgd

# Perform multi-run sweeps
python train.py -m optimizer.learning_rate=0.1,0.01,0.001 model=transformer,cnn
```

## Configuration Interpolation

Hydra allows you to reference other configuration values using the
`${path.to.value}` syntax:

```yaml
# model/transformer.yaml
name: transformer
embedding_size: 512
hidden_size: ${model.embedding_size}  # References embedding_size
```

## Environment Variables and Runtime Values

Access environment variables and compute values at runtime:

```yaml
data:
  path: ${oc.env:DATA_PATH,/default/path}  # Use env var or default
  num_workers: ${oc.select:device.type,cpu:4,gpu:8}  # Conditional value
```

## Advanced Features

### Optional Values

You can make configuration parameters optional using Python's `Optional` type:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    checkpoint_path: Optional[str] = None
    resume_training: bool = False
```

### Custom Validation

Add validation to your configuration using the `__post_init__` method:

```python
@dataclass
class OptimizerConfig:
    name: str = "adam"
    learning_rate: float = 0.001

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.learning_rate}")
```

### Configuration Composition

Combine multiple configurations using Hydra's composition system:

```yaml
# conf/config.yaml
defaults:
  - base_config
  - model: transformer
  - _self_  # Apply this config last
```

## Best Practices

1. **Use Type Hints**: Always include type hints for all configuration parameters.

2. **Set Sensible Defaults**: Provide reasonable default values to make your
   application usable with minimal configuration.

3. **Group Related Parameters**: Use nested dataclasses to organize related
   parameters logically.

4. **Document Parameters**: Add docstrings to your dataclasses and parameters
   to explain their purpose and valid values.

5. **Validate Configurations**: Add validation logic to catch invalid
   configurations early.

## Summary

HydraFlow's configuration system combines the type safety of Python dataclasses
with the flexibility of Hydra's composition and override capabilities. This
approach makes your machine learning experiments more maintainable,
reproducible, and easier to debug.