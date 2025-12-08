# pyright: basic
"""PyTorch implementation of Eckmann algorithm for multiple Lyapunov exponents.

This module provides GPU-accelerated computation of multiple Lyapunov exponents
using the Eckmann algorithm, vectorized for batch processing.

Reference:
    J. P. Eckmann, S. O. Kamphorst, D. Ruelle, and S. Ciliberto,
    "Liapunov exponents from time series," Physical Review A,
    vol. 34, no. 6, pp. 4971-4979, 1986.
"""

import torch
from torch import Tensor

from pybasin.feature_extractors.torch_feature_utilities import delay_embedding


@torch.no_grad()
def lyap_e_single(
    data: Tensor,
    emb_dim: int = 10,
    matrix_dim: int = 4,
    min_nb: int = 8,
    min_tsep: int = 0,
    tau: float = 1.0,
) -> Tensor:
    """Compute multiple Lyapunov exponents for a single 1D time series.

    Args:
        data: 1D time series of shape (N,)
        emb_dim: Embedding dimension (must satisfy (emb_dim-1) % (matrix_dim-1) == 0)
        matrix_dim: Matrix dimension for Jacobian estimation
        min_nb: Minimal number of neighbors
        min_tsep: Minimal temporal separation between neighbors
        tau: Time step size for normalization

    Returns:
        Tensor of matrix_dim Lyapunov exponents
    """
    # Handle unbounded trajectories
    if not torch.all(torch.isfinite(data)):
        return torch.full((matrix_dim,), float("inf"), dtype=data.dtype, device=data.device)

    m = (emb_dim - 1) // (matrix_dim - 1)

    # Build orbit (delay embedding)
    orbit = delay_embedding(data[:-m], emb_dim, lag=1)
    n_orbit = len(orbit)

    old_q = torch.eye(matrix_dim, device=data.device, dtype=data.dtype)
    lexp = torch.zeros(matrix_dim, device=data.device, dtype=data.dtype)
    lexp_counts = torch.zeros(matrix_dim, device=data.device, dtype=data.dtype)

    # Pre-compute stepped data
    step_offsets = torch.arange(matrix_dim, device=data.device) * m
    stepped_data = data[
        torch.arange(n_orbit, device=data.device).unsqueeze(1) + step_offsets.unsqueeze(0)
    ]
    beta_data = data[torch.arange(n_orbit, device=data.device) + matrix_dim * m]

    all_indices = torch.arange(n_orbit, device=data.device)

    for i in range(n_orbit):
        # Compute Chebyshev distances
        diffs = torch.max(torch.abs(orbit - orbit[i]), dim=1).values
        diffs[i] = float("inf")

        # Temporal masking
        time_mask = torch.abs(all_indices - i) <= min_tsep
        diffs = torch.where(time_mask, torch.tensor(float("inf"), device=data.device), diffs)

        # Find neighbors
        sorted_diffs, _ = torch.sort(diffs)
        r = sorted_diffs[min_nb - 1]

        if torch.isinf(r):
            continue

        neighbor_mask = diffs <= r
        n_neighbors = neighbor_mask.sum()

        if n_neighbors < min_nb:
            continue

        # Build least squares matrices
        orbit_i_stepped = stepped_data[i]
        beta_i = beta_data[i]

        x_diff = stepped_data - orbit_i_stepped
        beta_diff = beta_data - beta_i

        # Apply neighbor mask
        x_masked = x_diff[neighbor_mask]
        beta_masked = beta_diff[neighbor_mask]

        # Solve least squares using normal equations
        xtx = x_masked.T @ x_masked + 1e-10 * torch.eye(
            matrix_dim, device=data.device, dtype=data.dtype
        )
        xty = x_masked.T @ beta_masked

        try:
            a = torch.linalg.solve(xtx, xty)
        except RuntimeError:
            continue

        # Build T matrix
        t_mat = torch.zeros((matrix_dim, matrix_dim), device=data.device, dtype=data.dtype)
        t_mat[:-1, 1:] = torch.eye(matrix_dim - 1, device=data.device, dtype=data.dtype)
        t_mat[-1, :] = a

        # QR decomposition
        mat_q, mat_r = torch.linalg.qr(t_mat @ old_q)

        # Force positive diagonal
        sign_diag = torch.sign(torch.diag(mat_r))
        sign_diag = torch.where(sign_diag == 0, torch.ones_like(sign_diag), sign_diag)
        mat_q = mat_q @ torch.diag(sign_diag)
        mat_r = torch.diag(sign_diag) @ mat_r

        old_q = mat_q

        # Accumulate Lyapunov exponents
        diag_r = torch.diag(mat_r)
        pos_mask = diag_r > 0
        lexp[pos_mask] += torch.log(diag_r[pos_mask])
        lexp_counts[pos_mask] += 1

    # Normalize
    valid_mask = lexp_counts > 0
    lexp[valid_mask] /= lexp_counts[valid_mask]
    lexp[~valid_mask] = float("inf")
    lexp /= tau
    lexp /= m

    return lexp


@torch.no_grad()
def lyap_e_batch(
    data: Tensor,
    emb_dim: int = 10,
    matrix_dim: int = 4,
    min_nb: int = 8,
    min_tsep: int = 0,
    tau: float = 1.0,
) -> Tensor:
    """Compute multiple Lyapunov exponents for batch of multi-state trajectories.

    Args:
        data: Trajectories of shape (N, B, S) where:
            - N: number of time points
            - B: batch size (number of initial conditions)
            - S: number of state variables
        emb_dim: Embedding dimension
        matrix_dim: Matrix dimension
        min_nb: Minimal number of neighbors
        min_tsep: Minimal temporal separation
        tau: Time step size for normalization

    Returns:
        Array of Lyapunov exponents, shape (B, S, matrix_dim)
    """
    n_time, batch_size, n_states = data.shape

    # Transpose to (B, S, N)
    data_transposed = data.permute(1, 2, 0)

    results = torch.zeros(batch_size, n_states, matrix_dim, device=data.device)

    for b in range(batch_size):
        for s in range(n_states):
            series = data_transposed[b, s]
            results[b, s] = lyap_e_single(series, emb_dim, matrix_dim, min_nb, min_tsep, tau)

    return results
