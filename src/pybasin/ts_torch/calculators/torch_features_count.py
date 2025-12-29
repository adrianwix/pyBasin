# pyright: basic
import torch
from torch import Tensor

# =============================================================================
# COUNTING FEATURES (6 features)
# =============================================================================


@torch.no_grad()
def count_above(x: Tensor, t: float) -> Tensor:
    """Percentage of values above threshold t."""
    return (x > t).float().mean(dim=0)


@torch.no_grad()
def count_above_mean(x: Tensor) -> Tensor:
    """Count of values above mean."""
    mu = x.mean(dim=0, keepdim=True)
    return (x > mu).float().sum(dim=0)


@torch.no_grad()
def count_below(x: Tensor, t: float) -> Tensor:
    """Percentage of values below threshold t."""
    return (x < t).float().mean(dim=0)


@torch.no_grad()
def count_below_mean(x: Tensor) -> Tensor:
    """Count of values below mean."""
    mu = x.mean(dim=0, keepdim=True)
    return (x < mu).float().sum(dim=0)


@torch.no_grad()
def count_in_range(x: Tensor, min_val: float, max_val: float) -> Tensor:
    """Count of values in range [min_val, max_val]."""
    return ((x >= min_val) & (x <= max_val)).float().sum(dim=0)


@torch.no_grad()
def count_value(x: Tensor, value: float) -> Tensor:
    """Count of specific value."""
    return (x == value).float().sum(dim=0)


# =============================================================================
# BATCHED COUNTING FEATURES
# =============================================================================


@torch.no_grad()
def value_count_batched(x: Tensor, values: list[float]) -> Tensor:
    """Compute value_count for multiple values at once.

    Args:
        x: Input tensor of shape (N, B, S)
        values: List of values to count

    Returns:
        Tensor of shape (len(values), B, S)
    """
    batch_size, n_states = x.shape[1], x.shape[2]

    if not values:
        return torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device)

    results = []
    for value in values:
        results.append((x == value).float().sum(dim=0))

    return torch.stack(results, dim=0)


@torch.no_grad()
def range_count_batched(x: Tensor, params: list[dict]) -> Tensor:
    """Compute range_count for multiple (min_val, max_val) pairs at once.

    Args:
        x: Input tensor of shape (N, B, S)
        params: List of parameter dicts, each with keys "min_val" and "max_val"

    Returns:
        Tensor of shape (len(params), B, S)
    """
    batch_size, n_states = x.shape[1], x.shape[2]

    if not params:
        return torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device)

    results = []
    for p in params:
        min_val = p["min_val"]
        max_val = p["max_val"]
        results.append(((x >= min_val) & (x <= max_val)).float().sum(dim=0))

    return torch.stack(results, dim=0)
