# Feature Selectors

!!! note "Documentation in Progress"
This page is under construction.

## Overview

Feature selectors filter features before classification to remove constant, low-variance, or correlated features.

## Default Behavior

`DefaultFeatureSelector` removes features with zero variance.

## Using sklearn Transformers

```python
from sklearn.feature_selection import VarianceThreshold

bse = BasinStabilityEstimator(
    ...,
    feature_selector=VarianceThreshold(threshold=0.01),
)
```

## CorrelationSelector

```python
from pybasin.feature_selector.correlation_selector import CorrelationSelector

selector = CorrelationSelector(threshold=0.95)
```

## Disabling Feature Selection

```python
bse = BasinStabilityEstimator(..., feature_selector=None)
```
