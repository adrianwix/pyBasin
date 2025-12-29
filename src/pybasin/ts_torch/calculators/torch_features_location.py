# pyright: basic
import torch
from torch import Tensor

# =============================================================================
# LOCATION FEATURES (5 features)
# =============================================================================


@torch.no_grad()
def first_location_of_maximum(x: Tensor) -> Tensor:
    """Relative first location of maximum value."""
    n = x.shape[0]
    idx = x.argmax(dim=0)
    return idx.float() / n


@torch.no_grad()
def first_location_of_minimum(x: Tensor) -> Tensor:
    """Relative first location of minimum value."""
    n = x.shape[0]
    idx = x.argmin(dim=0)
    return idx.float() / n


@torch.no_grad()
def last_location_of_maximum(x: Tensor) -> Tensor:
    """Relative last location of maximum value."""
    n = x.shape[0]
    # Flip along time axis, find first max, compute last position
    x_flipped = x.flip(dims=[0])
    idx_from_end = x_flipped.argmax(dim=0)
    last_idx = n - 1 - idx_from_end
    return last_idx.float() / n


@torch.no_grad()
def last_location_of_minimum(x: Tensor) -> Tensor:
    """Relative last location of minimum value."""
    n = x.shape[0]
    min_val = x.min(dim=0, keepdim=True).values
    # Create indices tensor and mask where value equals min
    indices = torch.arange(n - 1, -1, -1, device=x.device).view(-1, 1, 1)
    is_min = x == min_val
    # Find last occurrence by finding first occurrence in reversed indices
    masked_idx = torch.where(is_min, indices.expand_as(x), torch.tensor(n, device=x.device))
    last_idx = n - 1 - masked_idx.min(dim=0).values
    return last_idx.float() / n


@torch.no_grad()
def index_mass_quantile(x: Tensor, q: float) -> Tensor:
    """Index where q% of cumulative mass is reached."""
    n = x.shape[0]
    x_abs = x.abs()
    cumsum = x_abs.cumsum(dim=0)
    total = cumsum[-1:]
    threshold = q * total
    # Find first index where cumsum >= threshold
    mask = cumsum >= threshold
    # Use argmax on mask to find first True
    idx = mask.float().argmax(dim=0)
    return idx.float() / n


# =============================================================================
# BATCHED LOCATION FEATURES
# =============================================================================


@torch.no_grad()
def index_mass_quantile_batched(x: Tensor, qs: list[float]) -> Tensor:
    """Compute index_mass_quantile for multiple q values at once.

    Args:
        x: Input tensor of shape (N, B, S)
        qs: List of quantile values

    Returns:
        Tensor of shape (len(qs), B, S)
    """
    n = x.shape[0]
    x_abs = x.abs()
    cumsum = x_abs.cumsum(dim=0)
    total = cumsum[-1:]

    results = []
    for q in qs:
        threshold = q * total
        mask = cumsum >= threshold
        idx = mask.float().argmax(dim=0)
        results.append(idx.float() / n)

    return torch.stack(results, dim=0)
