# Feature Extractors

!!! note "Documentation in Progress"
This page is under construction.

## Overview

Feature extractors compute time-series characteristics from trajectories that distinguish different attractor types.

## Base Class

All extractors inherit from `FeatureExtractor` and implement:

- Method: `extract_features(solution: Solution) -> torch.Tensor`
- Property: `feature_names`

## Available Extractors

| Class                     | Features | GPU | Speed   | Use Case                |
| ------------------------- | -------- | --- | ------- | ----------------------- |
| `TorchFeatureExtractor`   | ~700     | ✅  | Fast    | **Default**             |
| `JaxFeatureExtractor`     | ~50      | ✅  | Fastest | JAX-only workflows      |
| `TsFreshFeatureExtractor` | ~700     | ❌  | Slow    | Reference/validation    |
| `NoldsFeatureExtractor`   | ~10      | ❌  | Slow    | Dynamical features only |

## TorchFeatureExtractor (Default)

```python
from pybasin.feature_extractors import TorchFeatureExtractor

extractor = TorchFeatureExtractor(
    fc_parameters="minimal",  # or "comprehensive", custom dict
    time_steady=800.0,  # Discard transient
    device="cuda",
)
```

## Creating Custom Feature Extractors

See the [Custom Feature Extractor](../guides/custom-feature-extractor.md) guide.
