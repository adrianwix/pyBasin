# pyright: basic
"""Dynamical systems feature calculators for time series.

All feature functions follow a consistent tensor shape convention:
- Input: (N, B, S) where N=timesteps, B=batch size, S=state variables
- Output: (B, S) for scalar features, or (B, S, K) for multi-valued features where K is the number of values

Features are computed along the time dimension (dim=0), preserving batch and state dimensions.
"""

import torch
from torch import Tensor

from pybasin.ts_torch.torch_corr_dim import corr_dim_batch
from pybasin.ts_torch.torch_lyapunov_e import lyap_e_batch
from pybasin.ts_torch.torch_lyapunov_r import lyap_r_batch

# =============================================================================
# DYNAMICAL SYSTEMS FEATURES (5 features)
# =============================================================================


@torch.no_grad()
def lyapunov_r(
    x: Tensor,
    emb_dim: int = 10,
    lag: int = 1,
    trajectory_len: int = 20,
    tau: float = 1.0,
) -> Tensor:
    """Compute largest Lyapunov exponent using Rosenstein algorithm.

    :param x: Input time series tensor of shape (N, B, S) where N=timesteps, B=batch size, S=states.
    :param emb_dim: Embedding dimension for phase space reconstruction. Default is 10.
    :param lag: Lag for delay embedding. Default is 1.
    :param trajectory_len: Number of steps to follow divergence. Default is 20.
    :param tau: Time step size for normalization. Default is 1.0.
    :return: Tensor of shape (B, S) containing the largest Lyapunov exponent for each
        state of each batch.
    """
    return lyap_r_batch(x, emb_dim, lag, trajectory_len, tau)


@torch.no_grad()
def lyapunov_e(
    x: Tensor,
    emb_dim: int = 10,
    matrix_dim: int = 4,
    min_nb: int = 8,
    min_tsep: int = 0,
    tau: float = 1.0,
) -> Tensor:
    """Compute multiple Lyapunov exponents using Eckmann algorithm.

    :param x: Input time series tensor of shape (N, B, S) where N=timesteps, B=batch size, S=states.
    :param emb_dim: Embedding dimension for phase space reconstruction. Default is 10.
    :param matrix_dim: Matrix dimension for Jacobian estimation (number of exponents to compute).
        Default is 4.
    :param min_nb: Minimal number of neighbors required. Default is 8.
    :param min_tsep: Minimal temporal separation between neighbors. Default is 0.
    :param tau: Time step size for normalization. Default is 1.0.
    :return: Tensor of shape (B, S, matrix_dim) containing the Lyapunov exponents. The third
        dimension contains matrix_dim exponents sorted from largest to smallest.
    """
    return lyap_e_batch(x, emb_dim, matrix_dim, min_nb, min_tsep, tau)


@torch.no_grad()
def correlation_dimension(
    x: Tensor,
    emb_dim: int = 4,
    lag: int = 1,
    n_rvals: int = 50,
) -> Tensor:
    """Compute correlation dimension using Grassberger-Procaccia algorithm.

    :param x: Input time series tensor of shape (N, B, S) where N=timesteps, B=batch size, S=states.
    :param emb_dim: Embedding dimension for phase space reconstruction. Default is 4.
    :param lag: Lag for delay embedding. Default is 1.
    :param n_rvals: Number of radius values to use in correlation integral. Default is 50.
    :return: Tensor of shape (B, S) containing the correlation dimension for each state of each batch.
    """
    return corr_dim_batch(x, emb_dim, lag, n_rvals)


@torch.no_grad()
def friedrich_coefficients(x: Tensor, m: int = 3, r: float = 30.0, coeff: int = 0) -> Tensor:
    """Coefficients of polynomial fit to velocity vs position (fully batch vectorized)."""
    n, batch_size, n_states = x.shape

    # Compute velocity and position for all series
    velocity = x[1:] - x[:-1]  # (N-1, B, S)
    position = x[:-1]  # (N-1, B, S)

    # Reshape for batch processing: (B*S, N-1)
    velocity_flat = velocity.reshape(n - 1, -1).T  # (B*S, N-1)
    position_flat = position.reshape(n - 1, -1).T  # (B*S, N-1)

    # Build Vandermonde matrices: V[batch, time, power] = position^power
    # Powers: [m, m-1, ..., 1, 0]
    powers = torch.arange(m, -1, -1, device=x.device, dtype=x.dtype)  # (m+1,)
    V = position_flat.unsqueeze(-1) ** powers  # (B*S, N-1, m+1)

    # Solve least squares via normal equations: (V^T V) @ coeffs = V^T @ velocity
    # V^T @ V shape: (B*S, m+1, m+1)
    VtV = torch.bmm(V.transpose(1, 2), V)  # (B*S, m+1, m+1)
    # V^T @ velocity shape: (B*S, m+1)
    Vtv = torch.bmm(V.transpose(1, 2), velocity_flat.unsqueeze(-1)).squeeze(-1)  # (B*S, m+1)

    # Add regularization for numerical stability
    reg = 1e-6 * torch.eye(m + 1, device=x.device, dtype=x.dtype)
    VtV = VtV + reg

    # Solve using torch.linalg.solve (batched)
    try:
        coeffs = torch.linalg.solve(VtV, Vtv)  # (B*S, m+1)
    except Exception:
        # Fallback: use pseudo-inverse
        coeffs = torch.zeros(batch_size * n_states, m + 1, device=x.device, dtype=x.dtype)

    # Extract requested coefficient
    if coeff < m + 1:
        result = coeffs[:, coeff]
    else:
        result = torch.zeros(batch_size * n_states, device=x.device, dtype=x.dtype)

    return result.reshape(batch_size, n_states)


@torch.no_grad()
def max_langevin_fixed_point(x: Tensor, r: float = 3, m: int = 30) -> Tensor:
    """Maximum fixed point of Langevin model (fully batch vectorized)."""
    n, batch_size, n_states = x.shape
    m_int = int(m)

    # Compute velocity and position
    velocity = x[1:] - x[:-1]  # (N-1, B, S)
    position = x[:-1]  # (N-1, B, S)

    # Reshape for batch processing: (B*S, N-1)
    velocity_flat = velocity.reshape(n - 1, -1).T  # (B*S, N-1)
    position_flat = position.reshape(n - 1, -1).T  # (B*S, N-1)

    # Build Vandermonde matrices: V[batch, time, power] = position^power
    powers = torch.arange(m_int, -1, -1, device=x.device, dtype=x.dtype)  # (m+1,)
    vander = position_flat.unsqueeze(-1) ** powers  # (B*S, N-1, m+1)

    # Solve least squares via normal equations: (V^T V) @ coeffs = V^T @ velocity
    vtv = torch.bmm(vander.transpose(1, 2), vander)  # (B*S, m+1, m+1)
    vtv = vtv + 1e-6 * torch.eye(m_int + 1, device=x.device, dtype=x.dtype)  # Regularize
    vty = torch.bmm(vander.transpose(1, 2), velocity_flat.unsqueeze(-1)).squeeze(-1)  # (B*S, m+1)

    try:
        coeffs = torch.linalg.solve(vtv, vty)  # (B*S, m+1)
    except Exception:
        return torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device)

    # Compute fitted values: V @ coeffs
    fitted = torch.bmm(vander, coeffs.unsqueeze(-1)).squeeze(-1)  # (B*S, N-1)

    # Find zero crossings (sign changes)
    sign_change = fitted[:, :-1] * fitted[:, 1:] < 0  # (B*S, N-2)

    # Get position values at crossings, masked where no crossing
    position_at_crossings = position_flat[:, :-1]  # (B*S, N-2)
    position_at_crossings = torch.where(
        sign_change, position_at_crossings, torch.tensor(float("-inf"), device=x.device)
    )

    # Max position at crossings per series
    result = position_at_crossings.max(dim=1).values  # (B*S,)
    result = torch.where(result == float("-inf"), torch.zeros_like(result), result)

    return result.reshape(batch_size, n_states)


# =============================================================================
# BATCHED DYNAMICAL FEATURES
# =============================================================================


@torch.no_grad()
def friedrich_coefficients_batched(x: Tensor, params: list[dict]) -> Tensor:
    """Compute friedrich_coefficients for multiple coeff values at once.

    Groups by (m, r) combinations and computes polynomial fit once per group,
    then extracts all requested coefficients.

    Args:
        x: Input tensor of shape (N, B, S)
        params: List of parameter dicts, each with keys:
            - "m": int (polynomial degree)
            - "r": float (not used in computation, kept for API compatibility)
            - "coeff": int (coefficient index to extract)

    Returns:
        Tensor of shape (len(params), B, S)
    """
    n, batch_size, n_states = x.shape

    if not params:
        return torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device)

    velocity = x[1:] - x[:-1]
    position = x[:-1]
    velocity_flat = velocity.reshape(n - 1, -1).T
    position_flat = position.reshape(n - 1, -1).T

    groups: dict[tuple[int, float], list[tuple[int, int]]] = {}
    for idx, p in enumerate(params):
        key = (p["m"], p.get("r", 30.0))
        groups.setdefault(key, []).append((idx, p["coeff"]))

    results = torch.zeros(len(params), batch_size, n_states, dtype=x.dtype, device=x.device)

    for (m, _r), items in groups.items():
        powers = torch.arange(m, -1, -1, device=x.device, dtype=x.dtype)
        V = position_flat.unsqueeze(-1) ** powers

        VtV = torch.bmm(V.transpose(1, 2), V)
        Vtv = torch.bmm(V.transpose(1, 2), velocity_flat.unsqueeze(-1)).squeeze(-1)

        reg = 1e-6 * torch.eye(m + 1, device=x.device, dtype=x.dtype)
        VtV = VtV + reg

        try:
            coeffs = torch.linalg.solve(VtV, Vtv)
        except Exception:
            coeffs = torch.zeros(batch_size * n_states, m + 1, dtype=x.dtype, device=x.device)

        for param_idx, coeff in items:
            if coeff < m + 1:
                results[param_idx] = coeffs[:, coeff].reshape(batch_size, n_states)

    return results
