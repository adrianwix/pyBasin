# pyright: basic
"""PyTorch implementation of Grassberger-Procaccia algorithm for correlation dimension.

This module provides GPU-accelerated computation of correlation dimension
using the Grassberger-Procaccia algorithm, vectorized for batch processing.

Reference:
    P. Grassberger and I. Procaccia, "Characterization of strange
    attractors," Physical review letters, vol. 50, no. 5, p. 346, 1983.
"""

import torch
from torch import Tensor

from pybasin.feature_extractors.torch_feature_utilities import delay_embedding


def logarithmic_r(min_r: float, max_r: float, n_rvals: int, device: torch.device) -> Tensor:
    """Create logarithmically spaced radius values.

    Args:
        min_r: Minimum radius value
        max_r: Maximum radius value
        n_rvals: Number of values
        device: Device to create tensor on

    Returns:
        Tensor of logarithmically spaced values from min_r to max_r
    """
    log_min = torch.log(torch.tensor(min_r + 1e-10, device=device))
    log_max = torch.log(torch.tensor(max_r + 1e-10, device=device))
    return torch.exp(torch.linspace(log_min, log_max, n_rvals, device=device))


@torch.no_grad()
def corr_dim_single(
    data: Tensor,
    emb_dim: int = 4,
    lag: int = 1,
    n_rvals: int = 50,
) -> Tensor:
    """Compute correlation dimension for a single 1D time series.

    This is the Grassberger-Procaccia algorithm. The correlation dimension
    is a characteristic measure that describes the geometry of chaotic attractors.

    Args:
        data: 1D time series of shape (N,)
        emb_dim: Embedding dimension
        lag: Lag for delay embedding
        n_rvals: Number of radius values to use

    Returns:
        Correlation dimension (scalar)
    """
    # Create delay embedding
    orbit = delay_embedding(data, emb_dim, lag)
    n = orbit.shape[0]

    # Compute pairwise Euclidean distances
    dists = torch.cdist(orbit, orbit, p=2)

    # Zero out diagonal (self-distances)
    dists.fill_diagonal_(float("inf"))

    # Compute rvals based on data statistics
    sd = torch.std(data, correction=1)
    min_r = 0.1 * sd
    max_r = 0.5 * sd
    rvals = logarithmic_r(min_r.item(), max_r.item(), n_rvals, data.device)

    # Compute correlation sums for each radius
    csums = torch.zeros(n_rvals, device=data.device)
    for i, r in enumerate(rvals):
        count = (dists <= r).sum()
        csums[i] = count / (n * (n - 1))

    # Filter out zero csums and fit line in log-log space
    nonzero_mask = csums > 0
    log_r = torch.log(rvals)
    log_c = torch.where(
        nonzero_mask,
        torch.log(csums + 1e-30),
        torch.tensor(float("nan"), device=data.device),
    )

    # Linear regression: log(C) = D * log(r) + intercept
    n_valid = nonzero_mask.sum()

    log_r_valid = torch.where(nonzero_mask, log_r, torch.zeros_like(log_r))
    log_c_valid = torch.where(nonzero_mask, log_c, torch.zeros_like(log_c))

    sum_r = log_r_valid.sum()
    sum_c = log_c_valid.sum()
    sum_rr = (log_r_valid * log_r_valid).sum()
    sum_rc = (log_r_valid * log_c_valid).sum()

    denom = n_valid * sum_rr - sum_r * sum_r
    slope = torch.where(
        (n_valid >= 2) & (torch.abs(denom) > 1e-10),
        (n_valid * sum_rc - sum_r * sum_c) / denom,
        torch.tensor(float("nan"), device=data.device),
    )

    return slope


@torch.no_grad()
def corr_dim_batch(
    data: Tensor,
    emb_dim: int = 4,
    lag: int = 1,
    n_rvals: int = 50,
) -> Tensor:
    """Compute correlation dimension for batch of multi-state trajectories.

    Args:
        data: Trajectories of shape (N, B, S) where:
            - N: number of time points
            - B: batch size (number of initial conditions)
            - S: number of state variables
        emb_dim: Embedding dimension
        lag: Lag for delay embedding
        n_rvals: Number of radius values to use

    Returns:
        Array of correlation dimensions, shape (B, S)
    """
    n_time, batch_size, n_states = data.shape

    # Transpose to (B, S, N)
    data_transposed = data.permute(1, 2, 0)

    results = torch.zeros(batch_size, n_states, device=data.device)

    for b in range(batch_size):
        for s in range(n_states):
            series = data_transposed[b, s]
            results[b, s] = corr_dim_single(series, emb_dim, lag, n_rvals)

    return results
