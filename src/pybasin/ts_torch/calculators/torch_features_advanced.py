# pyright: basic
"""Advanced feature calculators that don't fit cleanly into other categories.

These features use specialized algorithms or test unique properties:
- benford_correlation: Tests first-digit distribution
- c3: Non-linearity measure using triple products
- energy_ratio_by_chunks: Temporal energy distribution
- time_reversal_asymmetry_statistic: Temporal asymmetry measure
"""

import math

import torch
from torch import Tensor

# =============================================================================
# ADVANCED FEATURES (4 features)
# =============================================================================


@torch.no_grad()
def benford_correlation(x: Tensor) -> Tensor:
    """Correlation with Benford's law distribution (vectorized)."""
    n, batch_size, n_states = x.shape
    benford = torch.tensor(
        [math.log10(1 + 1 / d) for d in range(1, 10)], dtype=x.dtype, device=x.device
    )
    benford_mean = benford.mean()
    benford_std = benford.std()
    benford_centered = benford - benford_mean

    result = torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device)

    # Process all series: get absolute values
    x_abs = x.abs()  # (N, B, S)

    # Mask for positive values
    pos_mask = x_abs > 0

    # Extract first digits for all values (will filter later)
    # first_digit = floor(x / 10^floor(log10(x)))
    log_x = torch.where(pos_mask, x_abs.log10(), torch.zeros_like(x_abs))
    floor_log = log_x.floor()
    first_digits = torch.where(
        pos_mask,
        (x_abs / (10**floor_log)).floor().long(),
        torch.zeros_like(x_abs, dtype=torch.long),
    )

    # Clamp to valid range and mask invalid
    valid_mask = pos_mask & (first_digits >= 1) & (first_digits <= 9)
    first_digits = first_digits.clamp(1, 9)  # (N, B, S)

    # Count frequencies for each batch/state using one-hot encoding
    # one_hot: (N, B, S, 9)
    one_hot = torch.nn.functional.one_hot(first_digits - 1, num_classes=9).float()
    one_hot = one_hot * valid_mask.unsqueeze(-1).float()  # Zero out invalid

    # Sum over time dimension: (B, S, 9)
    counts = one_hot.sum(dim=0)
    total_counts = counts.sum(dim=-1, keepdim=True)  # (B, S, 1)

    # Compute frequencies, handle zero counts
    freqs = counts / (total_counts + 1e-10)  # (B, S, 9)

    # Compute correlation with Benford's law
    freqs_mean = freqs.mean(dim=-1, keepdim=True)  # (B, S, 1)
    freqs_centered = freqs - freqs_mean  # (B, S, 9)
    freqs_std = freqs.std(dim=-1)  # (B, S)

    # Pearson correlation
    corr = (freqs_centered * benford_centered.unsqueeze(0).unsqueeze(0)).sum(dim=-1) / (
        freqs_std * benford_std * 9 + 1e-10
    )

    # Zero out results where we don't have enough valid data
    valid_counts = valid_mask.sum(dim=0)  # (B, S)
    result = torch.where(valid_counts >= 10, corr, torch.zeros_like(corr))

    return result


@torch.no_grad()
def c3(x: Tensor, lag: int) -> Tensor:
    """Non-linearity measure: mean(x[t] * x[t+lag] * x[t+2*lag])."""
    n = x.shape[0]
    if 2 * lag >= n:
        return torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)
    return (x[: -2 * lag] * x[lag:-lag] * x[2 * lag :]).mean(dim=0)


@torch.no_grad()
def energy_ratio_by_chunks(x: Tensor, num_segments: int = 10, segment_focus: int = 0) -> Tensor:
    """Energy ratio of a segment."""
    n = x.shape[0]
    segment_size = n // num_segments
    if segment_size == 0:
        return torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)

    total_energy = (x**2).sum(dim=0)
    start = segment_focus * segment_size
    end = min(start + segment_size, n)
    segment_energy = (x[start:end] ** 2).sum(dim=0)

    return segment_energy / (total_energy + 1e-10)


@torch.no_grad()
def time_reversal_asymmetry_statistic(x: Tensor, lag: int) -> Tensor:
    """Time reversal asymmetry statistic."""
    n = x.shape[0]
    if 2 * lag >= n:
        return torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)
    x_lag = x[lag:-lag]
    x_2lag = x[2 * lag :]
    x_0 = x[: -2 * lag]
    return (x_2lag**2 * x_lag - x_lag * x_0**2).mean(dim=0)


# =============================================================================
# BATCHED ADVANCED FEATURES
# =============================================================================


@torch.no_grad()
def energy_ratio_by_chunks_batched(
    x: Tensor, num_segments: int, segment_focuses: list[int]
) -> Tensor:
    """Compute energy ratio for multiple segment focuses at once.

    Args:
        x: Input tensor of shape (N, B, S)
        num_segments: Number of segments to divide the series into
        segment_focuses: List of segment indices to focus on

    Returns:
        Tensor of shape (len(segment_focuses), B, S)
    """
    n = x.shape[0]
    segment_size = n // num_segments
    if segment_size == 0:
        return torch.zeros(
            len(segment_focuses), x.shape[1], x.shape[2], dtype=x.dtype, device=x.device
        )

    total_energy = (x**2).sum(dim=0)

    results = []
    for segment_focus in segment_focuses:
        start = segment_focus * segment_size
        end = min(start + segment_size, n)
        segment_energy = (x[start:end] ** 2).sum(dim=0)
        results.append(segment_energy / (total_energy + 1e-10))

    return torch.stack(results, dim=0)


@torch.no_grad()
def c3_batched(x: Tensor, lags: list[int]) -> Tensor:
    """Compute c3 for multiple lag values at once.

    Args:
        x: Input tensor of shape (N, B, S)
        lags: List of lag values

    Returns:
        Tensor of shape (len(lags), B, S)
    """
    n, batch_size, n_states = x.shape

    if not lags:
        return torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device)

    results = []
    for lag in lags:
        if 2 * lag >= n:
            results.append(torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device))
        else:
            results.append((x[: -2 * lag] * x[lag:-lag] * x[2 * lag :]).mean(dim=0))

    return torch.stack(results, dim=0)


@torch.no_grad()
def time_reversal_asymmetry_statistic_batched(x: Tensor, lags: list[int]) -> Tensor:
    """Compute time_reversal_asymmetry_statistic for multiple lag values at once.

    Args:
        x: Input tensor of shape (N, B, S)
        lags: List of lag values

    Returns:
        Tensor of shape (len(lags), B, S)
    """
    n, batch_size, n_states = x.shape

    if not lags:
        return torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device)

    results = []
    for lag in lags:
        if 2 * lag >= n:
            results.append(torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device))
        else:
            x_lag = x[lag:-lag]
            x_2lag = x[2 * lag :]
            x_0 = x[: -2 * lag]
            results.append((x_2lag**2 * x_lag - x_lag * x_0**2).mean(dim=0))

    return torch.stack(results, dim=0)
