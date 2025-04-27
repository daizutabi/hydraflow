# Configuration Management

HydraFlow uses a powerful configuration management system based on Python's
dataclasses and Hydra's composition capabilities. This approach provides
type safety, IDE auto-completion, and flexible parameter specification.

## Basic Configuration

The simplest way to define a configuration for a HydraFlow application is
using a Python dataclass:

```python
from dataclasses import dataclass
from mlflow.entities import Run
import hydraflow

@dataclass
class Config:
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 10
    model_name: str = "transformer"

@hydraflow.main(Config)
def train(run: Run, cfg: Config) -> None:
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
def train(run: Run, cfg: Config) -> None:
    # Access nested configuration
    model_name = cfg.model.name
    lr = cfg.optimizer.learning_rate
    batch_size = cfg.data.batch_size
```

## Hydra Integration

HydraFlow integrates closely with Hydra for configuration management.
For detailed explanations of Hydra's capabilities, please refer to
the [Hydra documentation](https://hydra.cc/docs/intro/).

HydraFlow leverages the following Hydra features, but does not modify their behavior:

- **Configuration Files**: Organize configurations in YAML files
- **Command-line Overrides**: Change parameters without modifying code
- **Configuration Groups**: Swap entire configuration blocks
- **Configuration Composition**: Combine configurations from multiple sources
- **Interpolation**: Reference other configuration values
- **Multi-run Sweeps**: Run experiments with different parameter combinations

When using HydraFlow, remember that:

1. Your configuration structure comes from your dataclass definitions
2. HydraFlow automatically registers your top-level dataclass with Hydra
3. `@hydraflow.main` sets up the connection between your dataclass and Hydra

For advanced Hydra features and detailed usage examples, we recommend
consulting the official Hydra documentation after you become familiar
with the basic HydraFlow concepts.

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