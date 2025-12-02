"""JAX-based feature calculators for time series analysis.

This module provides GPU-accelerated feature calculators using JAX, matching
tsfresh's MinimalFCParameters. All functions are designed to work with
batched inputs and leverage JAX's vectorization and JIT compilation.

The calculators operate on time series data with shape (N, B, S) where:
- N: number of time steps
- B: batch size (number of trajectories)
- S: number of state variables

Each calculator returns features with shape (B, S).

Features implemented (matching tsfresh MinimalFCParameters):
- sum_values: Sum of all values
- median: Median value
- mean: Mean value
- length: Length of time series
- standard_deviation: Standard deviation
- variance: Variance
- root_mean_square: Root mean square
- maximum: Maximum value
- absolute_maximum: Maximum absolute value
- minimum: Minimum value
"""

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array


def sum_values(x: Array) -> Array:
    """Calculate the sum over the time series values."""
    return jnp.sum(x, axis=0)


def median(x: Array) -> Array:
    """Calculate the median of the time series."""
    return jnp.median(x, axis=0)


def mean(x: Array) -> Array:
    """Calculate the mean of the time series."""
    return jnp.mean(x, axis=0)


def length(x: Array) -> Array:
    """Calculate the length of the time series (number of time steps N)."""
    n = x.shape[0]
    return jnp.full(x.shape[1:], n, dtype=jnp.float32)  # pyright: ignore[reportUnknownMemberType]


def standard_deviation(x: Array) -> Array:
    """Calculate the standard deviation of the time series."""
    return jnp.std(x, axis=0)


def variance(x: Array) -> Array:
    """Calculate the variance of the time series."""
    return jnp.var(x, axis=0)


def root_mean_square(x: Array) -> Array:
    """Calculate the root mean square (RMS) of the time series."""
    return jnp.sqrt(jnp.mean(jnp.square(x), axis=0))


def maximum(x: Array) -> Array:
    """Calculate the maximum value of the time series."""
    return jnp.max(x, axis=0)


def absolute_maximum(x: Array) -> Array:
    """Calculate the maximum absolute value of the time series."""
    return jnp.max(jnp.abs(x), axis=0)


def minimum(x: Array) -> Array:
    """Calculate the minimum value of the time series."""
    return jnp.min(x, axis=0)


def delta(x: Array) -> Array:
    """Calculate the absolute difference between maximum and mean.

    This feature captures the spread of values around the mean and is useful
    for distinguishing between different dynamical behaviors:
    - Near-constant signals: delta ≈ 0
    - Oscillating signals: delta > 0
    """
    return jnp.abs(jnp.max(x, axis=0) - jnp.mean(x, axis=0))


def log_delta(x: Array) -> Array:
    """Calculate log(delta + epsilon) for improved feature space separation.

    Applies logarithmic transformation to the delta feature, which can
    linearize exponential ranges and improve classification performance
    when values span multiple orders of magnitude.
    """
    eps = 1e-12  # Small epsilon to avoid log(0)
    return jnp.log(delta(x) + eps)


# Dictionary of minimal features (matching tsfresh MinimalFCParameters - 10 features)
MINIMAL_FEATURES: dict[str, Callable[[Array], Array]] = {
    "sum_values": sum_values,
    "median": median,
    "mean": mean,
    "length": length,
    "standard_deviation": standard_deviation,
    "variance": variance,
    "root_mean_square": root_mean_square,
    "maximum": maximum,
    "absolute_maximum": absolute_maximum,
    "minimum": minimum,
}

# Dictionary of comprehensive features (minimal + custom features - 12 features)
COMPREHENSIVE_FEATURES: dict[str, Callable[[Array], Array]] = {
    **MINIMAL_FEATURES,
    "delta": delta,
    "log_delta": log_delta,
}

# All available features (for lookup by name)
ALL_FEATURES: dict[str, Callable[[Array], Array]] = COMPREHENSIVE_FEATURES


def get_feature_names(comprehensive: bool = False) -> list[str]:
    """Get the list of feature names.

    Args:
        comprehensive: If True, include custom features (delta, log_delta).
                      If False (default), return only tsfresh MinimalFCParameters.
    """
    if comprehensive:
        return list(COMPREHENSIVE_FEATURES.keys())
    return list(MINIMAL_FEATURES.keys())


def get_feature_functions(comprehensive: bool = False) -> dict[str, Callable[[Array], Array]]:
    """Get the feature functions dictionary.

    Args:
        comprehensive: If True, include custom features (delta, log_delta).
                      If False (default), return only tsfresh MinimalFCParameters.
    """
    if comprehensive:
        return COMPREHENSIVE_FEATURES.copy()
    return MINIMAL_FEATURES.copy()
