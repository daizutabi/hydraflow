# API Reference

This section provides detailed documentation for the public API of HydraFlow.
Use this reference to understand the capabilities and parameters of each
class and function.

## Core API

### Run

The `Run` class is the central abstraction in HydraFlow, representing a
single execution of a Hydra-configured application.

- [Run](./run.md) - Core class for working with individual runs
- [RunInfo](./run_info.md) - Metadata about a run

### Collections

Classes for working with groups of runs:

- [RunCollection](./run_collection.md) - Manipulating collections of runs
- [GroupedRuns](./grouped_runs.md) - Organizing runs into groups

## Analysis Tools

Tools for analyzing and visualizing run data:

- [Metrics](./metrics.md) - Functions for extracting and comparing metrics
- [Tables](./tables.md) - Creating tabular representations of run data
- [Plots](./plots.md) - Visualization utilities for run data

## Configuration

Utilities for working with configuration:

- [ConfigUtils](./config_utils.md) - Helper functions for configuration
- [Schema](./schema.md) - Configuration schema validation tools

## Extensions

Extension points for customizing HydraFlow:

- [Plugins](./plugins.md) - Plugin system for extending functionality
- [Hooks](./hooks.md) - Event hooks for custom behaviors
- [Serializers](./serializers.md) - Custom data serialization

## Utility Functions

Common utility functions:

- [File Operations](./file_ops.md) - Functions for working with files
- [Path Utilities](./path_utils.md) - Path manipulation utilities
- [Type Helpers](./type_helpers.md) - Type-related helper functions

## Environment

Environment configuration and management:

- [Environment Variables](./environment.md) - Available environment variables
- [Context](./context.md) - Context management utilities

## Integration

Integration with other libraries and tools:

- [MLflow Integration](./mlflow_integration.md) - Working with MLflow
- [Hydra Integration](./hydra_integration.md) - Working with Hydra

## Low-Level API

Low-level APIs (generally not needed for most users):

- [Implementation](./implementation.md) - Implementation classes
- [Internal Types](./internal_types.md) - Internal type definitions