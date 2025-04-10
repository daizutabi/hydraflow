# Visualizing Experiment Results

Visualizing experiment results is essential for understanding patterns,
comparing models, and communicating findings. HydraFlow's integration with
DataFrames makes it straightforward to create powerful visualizations using
popular Python libraries.

## Visualization Workflow

The general workflow for visualizing HydraFlow experiment results is:

1. **Load runs** using the [`Run`][hydraflow.core.run.Run] class
2. **Filter and prepare** data using [`RunCollection`][hydraflow.core.run_collection.RunCollection] methods
3. **Convert to DataFrame** using [`to_frame`][hydraflow.core.run_collection.RunCollection.to_frame]
4. **Create visualizations** using libraries like Matplotlib, Plotly, or Seaborn

## Basic Visualization with Matplotlib

[Matplotlib](https://matplotlib.org/) provides fundamental plotting capabilities:

```python
import matplotlib.pyplot as plt
from hydraflow import Run

# Load and prepare data
runs = Run.load("mlruns/0/*")
df = runs.to_frame(
    "learning_rate",
    "batch_size",
    accuracy=lambda run: run.metrics.get("accuracy", 0)
)

# Create a simple scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df["learning_rate"], df["accuracy"], s=df["batch_size"]/2)
plt.xscale("log")  # Log scale for learning rate
plt.xlabel("Learning Rate")
plt.ylabel("Accuracy")
plt.title("Model Accuracy vs Learning Rate")
plt.grid(True, alpha=0.3)
plt.savefig("accuracy_vs_lr.png", dpi=300)
```

## Interactive Visualizations with Plotly

[Plotly](https://plotly.com/python/) enables interactive visualizations:

```python
import plotly.express as px
import plotly.graph_objects as go

# Create an interactive scatter plot
fig = px.scatter(
    df.to_pandas(),  # Convert to pandas for plotly
    x="learning_rate",
    y="accuracy",
    size="batch_size",
    color="model_type",
    hover_data=["run_id", "epochs"],
    log_x=True,
    title="Model Performance by Hyperparameters"
)

# Customize the plot
fig.update_layout(
    xaxis_title="Learning Rate",
    yaxis_title="Accuracy",
    legend_title="Model Type"
)

# Save and show the plot
fig.write_html("interactive_results.html")
fig.show()
```

## Statistical Visualizations with Seaborn

[Seaborn](https://seaborn.pydata.org/) specializes in statistical visualizations:

```python
import seaborn as sns

# Set the style
sns.set(style="whitegrid")

# Create a boxplot by model type
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=df.to_pandas(),
    x="model_type",
    y="accuracy"
)
plt.title("Accuracy Distribution by Model Type")
plt.tight_layout()
plt.savefig("accuracy_distribution.png")

# Create a pair plot for multiple parameters
pair_df = df.select(["learning_rate", "batch_size", "dropout", "accuracy"]).to_pandas()
sns.pairplot(pair_df, hue="model_type", diag_kind="kde")
plt.savefig("parameter_relationships.png")
```

## Line Plots for Training Metrics

Visualize training progress with line plots:

```python
# Prepare data with epoch information
epochs_df = runs.to_frame(
    "model_type",
    "learning_rate",
    epochs=lambda run: list(range(len(run.metrics.get("epoch_accuracy", [])))),
    accuracy=lambda run: run.metrics.get("epoch_accuracy", [])
)

# Explode the epoch data (convert lists to rows)
import polars as pl
exploded_df = epochs_df.explode(["epochs", "accuracy"]).to_pandas()

# Plot training curves
plt.figure(figsize=(12, 6))
for model, group in exploded_df.groupby("model_type"):
    sns.lineplot(
        data=group,
        x="epochs",
        y="accuracy",
        label=model
    )
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Progress by Model Type")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("training_curves.png")
```

## Heatmaps for Parameter Studies

Create heatmaps to visualize the impact of hyperparameter combinations:

```python
# Prepare data for heatmap
heatmap_df = runs.group_by(
    "learning_rate",
    "batch_size",
    avg_accuracy=lambda runs: runs.mean("accuracy")
).to_pandas()

# Create a pivot table
pivot_df = heatmap_df.pivot(
    index="learning_rate",
    columns="batch_size",
    values="avg_accuracy"
)

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    pivot_df,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    cbar_kws={"label": "Average Accuracy"}
)
plt.title("Accuracy by Learning Rate and Batch Size")
plt.tight_layout()
plt.savefig("parameter_heatmap.png")
```

## Advanced Interactive Dashboards

For more complex analysis, create interactive dashboards with Plotly Dash:

```python
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Prepare your data
runs = Run.load("mlruns/0/*")
df = runs.to_frame(
    "model_type", "learning_rate", "batch_size", "optimizer_name",
    accuracy=lambda run: run.metrics.get("accuracy", 0),
    loss=lambda run: run.metrics.get("loss", float("inf"))
).to_pandas()

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define layout
app.layout = dbc.Container([
    html.H1("Experiment Analysis Dashboard"),

    dbc.Row([
        dbc.Col([
            html.Label("X-Axis:"),
            dcc.Dropdown(
                id='x-axis',
                options=[{'label': col, 'value': col} for col in df.columns],
                value='learning_rate'
            )
        ], width=4),

        dbc.Col([
            html.Label("Y-Axis:"),
            dcc.Dropdown(
                id='y-axis',
                options=[{'label': col, 'value': col} for col in df.columns],
                value='accuracy'
            )
        ], width=4),

        dbc.Col([
            html.Label("Color By:"),
            dcc.Dropdown(
                id='color-by',
                options=[{'label': col, 'value': col} for col in df.columns],
                value='model_type'
            )
        ], width=4)
    ], className="mb-4"),

    dcc.Graph(id='main-scatter-plot')
])

# Define callback
@app.callback(
    Output('main-scatter-plot', 'figure'),
    Input('x-axis', 'value'),
    Input('y-axis', 'value'),
    Input('color-by', 'value')
)
def update_graph(x_column, y_column, color_column):
    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=color_column,
        size='batch_size',
        hover_data=['run_id', 'optimizer_name']
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
```

## Visualizing Model Performance

Create specialized visualizations for model evaluation:

```python
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

# Function to get evaluation metrics from a run
def get_eval_data(run):
    artifacts_dir = run.info.run_dir / "artifacts"
    try:
        y_true = np.load(artifacts_dir / "y_true.npy")
        y_pred = np.load(artifacts_dir / "y_pred.npy")
        y_score = np.load(artifacts_dir / "y_score.npy")
        return y_true, y_pred, y_score
    except FileNotFoundError:
        return None, None, None

# Get a specific run
run = runs.get(model_type="transformer", learning_rate=0.001)
y_true, y_pred, y_score = get_eval_data(run)

if y_true is not None:
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("roc_curve.png")
```

## Comparing Multiple Runs

Visualize performance across multiple runs:

```python
# Extract data for all runs
run_data = []
for run in runs:
    run_data.append({
        "run_id": run.info.run_id,
        "model_type": run.get("model_type"),
        "learning_rate": run.get("learning_rate"),
        "accuracy": run.metrics.get("accuracy", 0),
        "precision": run.metrics.get("precision", 0),
        "recall": run.metrics.get("recall", 0),
        "f1": run.metrics.get("f1", 0)
    })

# Convert to DataFrame
import pandas as pd
performance_df = pd.DataFrame(run_data)

# Create a parallel coordinates plot
fig = px.parallel_coordinates(
    performance_df,
    color="accuracy",
    dimensions=["learning_rate", "accuracy", "precision", "recall", "f1"],
    color_continuous_scale=px.colors.sequential.Viridis,
    title="Model Performance Metrics Across Runs"
)
fig.show()
```

## Visualizing Hyperparameter Importance

Assess the impact of different hyperparameters:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# Prepare data
X = df[["learning_rate", "batch_size", "dropout", "hidden_size"]].to_numpy()
y = df["accuracy"].to_numpy()

# Train a random forest to model hyperparameter importance
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
feature_names = ["learning_rate", "batch_size", "dropout", "hidden_size"]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Importance")
plt.title("Hyperparameter Importance")
plt.tight_layout()
plt.savefig("hyperparameter_importance.png")
```

## Best Practices

1. **Choose the Right Visualization**: Select visualizations that best convey
   the patterns and relationships in your data.

2. **Ensure Consistency**: Use consistent color schemes and styles across
   related visualizations.

3. **Add Context**: Include proper titles, axis labels, and legends to make
   visualizations self-explanatory.

4. **Interactive vs. Static**: Choose interactive visualizations for
   exploration and static ones for reports and publications.

5. **Preprocessing Data**: Clean and transform your data appropriately before
   visualization to avoid misleading representations.

## Summary

Visualizing experiment results is a crucial step in the machine learning
workflow. HydraFlow's integration with Polars DataFrames makes it easy to
transform experiment data into formats suitable for visualization with
libraries like Matplotlib, Plotly, and Seaborn. These visualizations help
you understand model performance, identify optimal hyperparameters, and
communicate results effectively.