# pyright: basic

"""PyTorch utility functions for feature extraction.

This module provides utility functions for handling NaN, inf, and other
edge cases in feature arrays, similar to tsfresh's dataframe_functions.
"""

import torch
from torch import Tensor


def impute(features: Tensor) -> Tensor:
    """
    Columnwise replaces all NaNs and infs from the feature array with average/extreme values.

    This is done as follows for each column:
        * -inf -> min (of finite values in that column)
        * +inf -> max (of finite values in that column)
        * NaN -> median (of finite values in that column)

    If a column does not contain any finite values at all, it is filled with zeros.

    This function is the PyTorch equivalent of tsfresh's impute function.

    Parameters
    ----------
    features : Tensor
        Feature tensor of shape (B, F) where B is batch size and F is number of features.

    Returns
    -------
    Tensor
        Imputed feature tensor with the same shape, guaranteed to contain no NaN or inf values.
    """
    result = features.clone()
    n_cols = features.shape[1]

    for col_idx in range(n_cols):
        col = features[:, col_idx]

        finite_mask = torch.isfinite(col)
        finite_vals = col[finite_mask]

        if finite_vals.numel() == 0:
            result[:, col_idx] = 0.0
            continue

        col_min = finite_vals.min()
        col_max = finite_vals.max()
        col_median = finite_vals.median()

        neg_inf_mask = torch.isneginf(col)
        pos_inf_mask = torch.isposinf(col)
        nan_mask = torch.isnan(col)

        result[neg_inf_mask, col_idx] = col_min
        result[pos_inf_mask, col_idx] = col_max
        result[nan_mask, col_idx] = col_median

    return result


def impute_extreme(features: Tensor, extreme_value: float = 1e10) -> Tensor:
    """
    Replaces all NaNs and infs with extreme values to make them distinguishable.

    This is useful when you want samples with non-finite features to be
    classified separately (e.g., unbounded trajectories in dynamical systems).

    - +inf -> +extreme_value
    - -inf -> -extreme_value
    - NaN -> +extreme_value (to cluster with +inf)

    Parameters
    ----------
    features : Tensor
        Feature tensor of shape (B, F) where B is batch size and F is number of features.
    extreme_value : float
        The extreme value to use for replacement. Default is 1e10.

    Returns
    -------
    Tensor
        Imputed feature tensor with the same shape.
    """
    result = features.clone()
    result = torch.where(
        torch.isposinf(features),
        torch.tensor(extreme_value, device=features.device, dtype=features.dtype),
        result,
    )
    result = torch.where(
        torch.isneginf(features),
        torch.tensor(-extreme_value, device=features.device, dtype=features.dtype),
        result,
    )
    result = torch.where(
        torch.isnan(features),
        torch.tensor(extreme_value, device=features.device, dtype=features.dtype),
        result,
    )
    return result


def delay_embedding(data: Tensor, emb_dim: int, lag: int = 1) -> Tensor:
    """Create delay embedding of a time series.

    Args:
        data: Time series of shape (N,) for 1D or (N, ...) for higher dims
        emb_dim: Embedding dimension
        lag: Lag between elements in embedded vectors

    Returns:
        Embedded data of shape (M, emb_dim) for 1D input where M = N - (emb_dim-1)*lag
    """
    n = data.shape[0]
    m = n - (emb_dim - 1) * lag
    indices = torch.arange(emb_dim, device=data.device) * lag
    indices = indices.unsqueeze(0) + torch.arange(m, device=data.device).unsqueeze(1)
    return data[indices]


def rowwise_euclidean(x: Tensor, y: Tensor) -> Tensor:
    """Compute Euclidean distance from each row of x to vector y.

    Args:
        x: Matrix of shape (M, D)
        y: Vector of shape (D,)

    Returns:
        Distances of shape (M,)
    """
    return torch.norm(x - y, dim=-1)


def rowwise_chebyshev(x: Tensor, y: Tensor) -> Tensor:
    """Compute Chebyshev (L-infinity) distance from each row of x to vector y.

    Args:
        x: Matrix of shape (M, D)
        y: Vector of shape (D,)

    Returns:
        Distances of shape (M,)
    """
    return torch.max(torch.abs(x - y), dim=1).values


@torch.no_grad()
def local_maxima_1d(x: Tensor) -> Tensor:
    """Find local maxima in batched 1D signals.

    This is a vectorized PyTorch implementation equivalent to scipy's _local_maxima_1d.
    It finds all local maxima (including plateaus) and returns a boolean mask with True
    at the midpoint of each maximum.

    A maximum is defined as one or more samples of equal value that are surrounded
    on both sides by at least one smaller sample. For plateaus, the midpoint
    (rounded down for even sizes) is marked as the peak.

    Args:
        x: Input tensor of shape (N, B, S) where N is timesteps, B is batch size,
           and S is number of states/signals.

    Returns:
        Boolean mask of shape (N, B, S) with True at peak positions (midpoints for plateaus).
        First and last samples are always False (cannot be maxima by definition).
    """
    n_timesteps, batch_size, n_states = x.shape
    device = x.device

    peaks_mask = torch.zeros(n_timesteps, batch_size, n_states, dtype=torch.bool, device=device)

    if n_timesteps < 3:
        return peaks_mask

    rising = x[:-1] < x[1:]
    falling = x[:-1] > x[1:]
    equal = x[:-1] == x[1:]

    rising_edge = torch.zeros(n_timesteps, batch_size, n_states, dtype=torch.bool, device=device)
    rising_edge[1:] = rising

    falling_edge = torch.zeros(n_timesteps, batch_size, n_states, dtype=torch.bool, device=device)
    falling_edge[:-1] = falling

    equal_pad = torch.zeros(n_timesteps, batch_size, n_states, dtype=torch.bool, device=device)
    equal_pad[:-1] = equal

    plateau_start = rising_edge & ~torch.roll(rising_edge, 1, dims=0)
    plateau_start[0] = False

    in_plateau = torch.zeros_like(peaks_mask)
    for t in range(1, n_timesteps - 1):
        still_equal = equal_pad[t]
        was_rising = rising_edge[t]
        continues = in_plateau[t - 1] & still_equal
        in_plateau[t] = was_rising | continues

    plateau_end = in_plateau & falling_edge
    plateau_end[-1] = False

    flat_x = x.reshape(n_timesteps, -1)
    flat_peaks = peaks_mask.reshape(n_timesteps, -1)

    for signal_idx in range(flat_x.shape[1]):
        signal = flat_x[:, signal_idx]
        i = 1
        i_max = n_timesteps - 1

        while i < i_max:
            if signal[i - 1] < signal[i]:
                i_ahead = i + 1
                while i_ahead < i_max and signal[i_ahead] == signal[i]:
                    i_ahead += 1
                if signal[i_ahead] < signal[i]:
                    midpoint = (i + i_ahead - 1) // 2
                    flat_peaks[midpoint, signal_idx] = True
                    i = i_ahead
            i += 1

    return flat_peaks.reshape(n_timesteps, batch_size, n_states)
