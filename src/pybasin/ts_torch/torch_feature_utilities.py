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

    :param features: Feature tensor of shape (B, F) where:
        - B: batch size (number of samples/trajectories)
        - F: number of features
    :return: Imputed feature tensor of shape (B, F), guaranteed to contain no NaN or inf values.
        Each column is imputed independently using the strategy described above.
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

    :param features: Feature tensor of shape (B, F) where:
        - B: batch size (number of samples/trajectories)
        - F: number of features
    :param extreme_value: The extreme value to use for replacement. Default is 1e10.
    :return: Imputed feature tensor of shape (B, F). All inf and NaN values are replaced
        with ±extreme_value, allowing unbounded trajectories to be distinguished in clustering.
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

    Constructs a matrix of delayed vectors from a 1D time series, useful for phase space
    reconstruction in dynamical systems analysis.

    :param data: Time series tensor. Can be:
        - 1D: shape (N,) where N is the number of time points
        - Higher dimensional: shape (N, ...) where additional dimensions are preserved
    :param emb_dim: Embedding dimension. Determines how many delayed copies of the signal
        are concatenated to form each embedding vector.
    :param lag: Lag (delay) between elements in embedded vectors. Specifies the time delay
        in units of the sampling interval. Default is 1.
    :return: Embedded data tensor of shape (M, emb_dim) for 1D input, where
        M = N - (emb_dim-1)*lag is the number of embedded vectors. Each row contains
        [data[i], data[i+lag], data[i+2*lag], ..., data[i+(emb_dim-1)*lag]].
    """
    n = data.shape[0]
    m = n - (emb_dim - 1) * lag
    indices = torch.arange(emb_dim, device=data.device) * lag
    indices = indices.unsqueeze(0) + torch.arange(m, device=data.device).unsqueeze(1)
    return data[indices]


def rowwise_euclidean(x: Tensor, y: Tensor) -> Tensor:
    """Compute Euclidean distance from each row of x to vector y.

    :param x: Matrix of shape (M, D) where M is the number of vectors and D is the dimension.
    :param y: Vector of shape (D,) to compute distances from.
    :return: Tensor of shape (M,) containing the Euclidean (L2) distance from each row
        of x to vector y.
    """
    return torch.norm(x - y, dim=-1)


def rowwise_chebyshev(x: Tensor, y: Tensor) -> Tensor:
    """Compute Chebyshev (L-infinity) distance from each row of x to vector y.

    The Chebyshev distance is the maximum absolute difference across all dimensions.

    :param x: Matrix of shape (M, D) where M is the number of vectors and D is the dimension.
    :param y: Vector of shape (D,) to compute distances from.
    :return: Tensor of shape (M,) containing the Chebyshev (L-infinity) distance from each
        row of x to vector y. Each distance is max(|x[i] - y|) across all dimensions.
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
