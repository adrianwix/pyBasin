# Predictors

!!! note "Documentation in Progress"
This page is under construction.

## Overview

Predictors classify trajectories into attractor classes based on extracted features.

## Predictor Types

- **Unsupervised** (`ClustererPredictor`): Discovers attractor classes automatically
- **Supervised** (`ClassifierPredictor`): Uses known template trajectories

## Available Predictors

| Class                    | Type         | Description                                       |
| ------------------------ | ------------ | ------------------------------------------------- |
| `HDBSCANClusterer`       | Unsupervised | **Default**, density-based, auto-tunes parameters |
| `DBSCANClusterer`        | Unsupervised | Classic DBSCAN                                    |
| `DynamicalClusterer`     | Unsupervised | Physics-based two-stage clustering                |
| `KNNClassifier`          | Supervised   | K-nearest neighbors with templates                |
| `UnboundednessClusterer` | Hybrid       | Detects unbounded trajectories                    |

## HDBSCANClusterer (Default)

```python
from pybasin.predictors.hdbscan_clusterer import HDBSCANClusterer

predictor = HDBSCANClusterer(
    auto_tune=True,      # Auto-select min_cluster_size
    assign_noise=True,   # Assign noise points to nearest cluster
    min_cluster_size=50, # If auto_tune=False
)
```

## KNNClassifier (Supervised)

```python
from pybasin.predictors.knn_classifier import KNNClassifier

predictor = KNNClassifier(
    template_y0=[[0.0, 0.0], [1.0, 0.0]],  # Template ICs
    labels=["FP", "LC"],
    k=5,
)
```
