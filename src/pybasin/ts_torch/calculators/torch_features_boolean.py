# pyright: basic
import torch
from torch import Tensor

# =============================================================================
# BOOLEAN FEATURES (5 features)
# =============================================================================


@torch.no_grad()
def has_duplicate(x: Tensor) -> Tensor:
    """Check if any value occurs more than once (optimized with sorting)."""
    n, batch_size, n_states = x.shape
    # Reshape to (N, B*S) and sort along time dimension
    x_flat = x.reshape(n, -1)
    sorted_x, _ = x_flat.sort(dim=0)
    # Check if any adjacent sorted values are equal
    has_adj_dup = (sorted_x[1:] == sorted_x[:-1]).any(dim=0)  # (B*S,)
    return has_adj_dup.float().reshape(batch_size, n_states)


@torch.no_grad()
def has_duplicate_max(x: Tensor) -> Tensor:
    """Check if maximum value occurs more than once."""
    max_val = x.max(dim=0, keepdim=True).values
    return ((x == max_val).float().sum(dim=0) > 1).float()


@torch.no_grad()
def has_duplicate_min(x: Tensor) -> Tensor:
    """Check if minimum value occurs more than once."""
    min_val = x.min(dim=0, keepdim=True).values
    return ((x == min_val).float().sum(dim=0) > 1).float()


@torch.no_grad()
def has_variance_larger_than_standard_deviation(x: Tensor) -> Tensor:
    """Check if variance > standard deviation (equivalent to std > 1)."""
    std = x.std(dim=0, correction=0)
    return (std > 1).float()


@torch.no_grad()
def has_large_standard_deviation(x: Tensor, r: float = 0.25) -> Tensor:
    """Check if std > r * range."""
    std = x.std(dim=0, correction=0)
    range_val = x.max(dim=0).values - x.min(dim=0).values
    return (std > r * range_val).float()
