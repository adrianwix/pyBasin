# pyright: basic
"""PyTorch implementation of Rosenstein algorithm for largest Lyapunov exponent.

This module provides GPU-accelerated computation of the largest Lyapunov exponent
using the Rosenstein algorithm, vectorized for batch processing.

Reference:
    M. T. Rosenstein, J. J. Collins, and C. J. De Luca,
    "A practical method for calculating largest Lyapunov exponents from
    small data sets," Physica D: Nonlinear Phenomena, vol. 65, no. 1,
    pp. 117-134, 1993.
"""

import torch
from torch import Tensor

from pybasin.feature_extractors.torch_feature_utilities import delay_embedding


def compute_min_tsep(data: Tensor, n: int) -> Tensor:
    """Compute minimum temporal separation using mean frequency.

    Args:
        data: 1D time series
        n: Length of data

    Returns:
        Minimum temporal separation
    """
    max_tsep_factor = 0.25

    f = torch.fft.rfft(data, n=n * 2 - 1)
    freqs = torch.fft.rfftfreq(n * 2 - 1, device=data.device)
    psd = torch.abs(f) ** 2

    mf = torch.sum(freqs[1:] * psd[1:]) / (torch.sum(psd[1:]) + 1e-10)
    min_tsep = torch.ceil(1.0 / (mf + 1e-10)).long()

    max_tsep = int(max_tsep_factor * n)
    min_tsep = torch.clamp(min_tsep, min=1, max=max_tsep)

    return min_tsep


def _poly_line_fit(ks: Tensor, div_traj: Tensor, finite_mask: Tensor) -> Tensor:
    """Fit a line using least squares.

    Args:
        ks: x-values (k indices), shape (N,)
        div_traj: y-values (divergence trajectory), shape (N,)
        finite_mask: Boolean mask for valid points, shape (N,)

    Returns:
        Slope of the fitted line
    """
    n_finite = finite_mask.sum()

    ks_finite = torch.where(finite_mask, ks, torch.zeros_like(ks))
    div_finite = torch.where(finite_mask, div_traj, torch.zeros_like(div_traj))

    sum_k = ks_finite.sum()
    sum_d = div_finite.sum()
    sum_kk = (ks_finite * ks_finite).sum()
    sum_kd = (ks_finite * div_finite).sum()

    denom = n_finite * sum_kk - sum_k * sum_k
    slope = torch.where(
        (n_finite >= 2) & (torch.abs(denom) > 1e-10),
        (n_finite * sum_kd - sum_k * sum_d) / denom,
        torch.tensor(float("nan"), device=ks.device),
    )

    return slope


@torch.no_grad()
def lyap_r_single(
    data: Tensor,
    emb_dim: int = 10,
    lag: int = 1,
    trajectory_len: int = 20,
    tau: float = 1.0,
) -> Tensor:
    """Compute largest Lyapunov exponent for a single 1D time series.

    Args:
        data: 1D time series of shape (N,)
        emb_dim: Embedding dimension
        lag: Lag for delay embedding
        trajectory_len: Number of steps to follow divergence
        tau: Time step size for normalization

    Returns:
        Largest Lyapunov exponent (scalar)
    """
    n = data.shape[0]
    min_tsep = compute_min_tsep(data, n)

    orbit = delay_embedding(data, emb_dim, lag)
    m = orbit.shape[0]

    # Compute pairwise distances
    dists = torch.cdist(orbit, orbit, p=2)

    # Mask self and temporally close neighbors
    mask_range = torch.arange(m, device=data.device)
    for_masking = torch.abs(mask_range.unsqueeze(1) - mask_range.unsqueeze(0)) <= min_tsep
    dists = torch.where(for_masking, torch.tensor(float("inf"), device=data.device), dists)

    ntraj = max(m - trajectory_len + 1, 1)

    # Find nearest neighbors for each point
    search_dists = dists[:ntraj, :ntraj].clone()
    nb_idx = torch.argmin(search_dists, dim=1)

    # Compute divergence trajectory
    div_traj = torch.zeros(trajectory_len, device=data.device)
    for k in range(trajectory_len):
        i_indices = torch.arange(ntraj, device=data.device) + k
        j_indices = nb_idx + k

        valid_i = i_indices < m
        valid_j = j_indices < m
        valid = valid_i & valid_j

        div_k = torch.where(
            valid,
            dists[i_indices.clamp(max=m - 1), j_indices.clamp(max=m - 1)],
            torch.tensor(float("nan"), device=data.device),
        )

        div_k_safe = torch.where((div_k > 0) & valid, div_k, torch.ones_like(div_k))
        log_div_k = torch.log(div_k_safe)
        log_div_k = torch.where(
            (div_k > 0) & valid, log_div_k, torch.tensor(float("nan"), device=data.device)
        )

        div_traj[k] = torch.nanmean(log_div_k)

    # Fit line to divergence trajectory
    ks = torch.arange(trajectory_len, dtype=torch.float32, device=data.device)
    finite_mask = torch.isfinite(div_traj)

    slope = _poly_line_fit(ks, div_traj, finite_mask)
    le = slope / tau

    return le


@torch.no_grad()
def lyap_r_batch(
    data: Tensor,
    emb_dim: int = 10,
    lag: int = 1,
    trajectory_len: int = 20,
    tau: float = 1.0,
) -> Tensor:
    """Compute Lyapunov exponents for batch of multi-state trajectories.

    Args:
        data: Trajectories of shape (N, B, S) where:
            - N: number of time points
            - B: batch size (number of initial conditions)
            - S: number of state variables
        emb_dim: Embedding dimension
        lag: Lag for delay embedding
        trajectory_len: Number of steps to follow divergence
        tau: Time step size for normalization

    Returns:
        Array of Lyapunov exponents, shape (B, S)
    """
    n_time, batch_size, n_states = data.shape

    # Transpose to (B, S, N)
    data_transposed = data.permute(1, 2, 0)

    results = torch.zeros(batch_size, n_states, device=data.device)

    for b in range(batch_size):
        for s in range(n_states):
            series = data_transposed[b, s]
            results[b, s] = lyap_r_single(series, emb_dim, lag, trajectory_len, tau)

    return results
