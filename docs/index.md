<style>
  h1 {
    display: none;
  }
  .md-main__inner {
    margin-top: 0;
  }
  .hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 2rem 1rem;
    background-color: #f5f5f5;
  }
  .hero-title {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 1rem;
  }
  .hero-subtitle {
    font-size: 1.2rem;
    margin-bottom: 2rem;
    max-width: 600px;
  }
  .hero-buttons .md-button {
    margin: 0.5rem;
  }
  [data-md-color-scheme="slate"] .hero {
    background-color: #2c2c2c;
  }
</style>

<div class="hero">
  <div class="hero-title">HydraFlow</div>
  <div class="hero-subtitle">Seamlessly integrate Hydra and MLflow to streamline machine learning experiment workflows.</div>
  <div class="hero-buttons">
    <a href="getting-started/" class="md-button md-button--primary">
      Getting Started
    </a>
    <a href="https://github.com/daizutabi/hydraflow" class="md-button">
      View on GitHub
    </a>
  </div>
</div>

## What is HydraFlow?

HydraFlow seamlessly integrates [Hydra](https://hydra.cc/) and
[MLflow](https://mlflow.org/) to create a comprehensive machine learning
experiment management framework. It provides a complete workflow from defining
experiments to execution and analysis, streamlining machine learning projects
from research to production.

### Key Integration Features

- :dart: **Automatic Configuration Tracking**: Hydra configurations are automatically
  saved as MLflow artifacts, ensuring complete reproducibility of experiments.
- :lock: **Type-safe Configuration**: Leverage Python dataclasses for type-safe
  experiment configuration with full IDE support.
- :recycle: **Unified Workflow**: Connect configuration management and experiment tracking
  in a single, coherent workflow.
- :chart_with_upwards_trend: **Powerful Analysis Tools**: Analyze and compare experiments using
  configuration parameters captured from Hydra.

## Where to go next?

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **Getting Started**

    ---

    Install HydraFlow and learn the core concepts and design principles to get you up and running quickly.

    [:octicons-arrow-right-24: Learn the basics](getting-started/index.md)

-   :material-school:{ .lg .middle } **Practical Tutorials**

    ---

    Follow hands-on examples to understand HydraFlow in practice and learn to create your first application.

    [:octicons-arrow-right-24: Try the tutorials](practical-tutorials/index.md)

-   :material-tools:{ .lg .middle } **User Guide**

    ---

    Dive deep into the features of HydraFlow, from running applications to automating workflows and analyzing results.

    [:octicons-arrow-right-24: Read the guide](part1-applications/index.md)

</div>
