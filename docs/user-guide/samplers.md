# Samplers

!!! note "Documentation in Progress"
This page is under construction.

## Overview

Samplers generate initial conditions for basin stability estimation.

## Base Class

All samplers inherit from `Sampler` and implement:

```python
def sample(n: int) -> torch.Tensor
```

## Available Samplers

| Class                  | Description                 | Use Case                        |
| ---------------------- | --------------------------- | ------------------------------- |
| `UniformRandomSampler` | Uniform random in hypercube | General purpose, most common    |
| `GridSampler`          | Evenly spaced grid          | 2D visualization, deterministic |

## UniformRandomSampler

```python
from pybasin.sampler import UniformRandomSampler

sampler = UniformRandomSampler(
    min_limits=[-np.pi, -2.0],
    max_limits=[np.pi, 2.0],
    device="cuda",  # optional
)
```

## GridSampler

```python
from pybasin.sampler import GridSampler

sampler = GridSampler(
    min_limits=[-np.pi, -2.0],
    max_limits=[np.pi, 2.0],
    fixed_dims={2: 0.0},  # Fix 3rd dimension to 0
)
```

## Creating Custom Samplers

Inherit from `Sampler` and implement the `sample(n: int) -> torch.Tensor` method.
