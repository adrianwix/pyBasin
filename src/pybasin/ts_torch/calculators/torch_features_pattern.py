# pyright: basic
import torch
from torch import Tensor

# =============================================================================
# STREAK/PATTERN FEATURES (5 features)
# =============================================================================


def _longest_consecutive_run(mask: Tensor) -> Tensor:
    """Helper to find longest consecutive True run along dim 0 (vectorized)."""
    # mask: (N, B, S) boolean tensor
    n, batch_size, n_states = mask.shape

    if n == 0:
        return torch.zeros(batch_size, n_states, dtype=torch.float32, device=mask.device)

    # Convert to float for arithmetic
    m = mask.float()  # (N, B, S)

    # For each position, compute cumulative run length
    # This is equivalent to: for each True, count consecutive Trues before it
    result = torch.zeros(batch_size, n_states, dtype=torch.float32, device=mask.device)

    # Use cumsum with reset trick: multiply by mask to reset on False
    # cumsum of mask, but reset when mask is False
    # We compute run lengths by: if mask[i], run[i] = run[i-1] + 1, else 0
    run_lengths = torch.zeros_like(m)
    current_run = torch.zeros(batch_size, n_states, device=mask.device)

    for i in range(n):
        current_run = (current_run + 1) * m[i]  # Reset to 0 when False, else increment
        run_lengths[i] = current_run

    result = run_lengths.max(dim=0).values
    return result


@torch.no_grad()
def longest_strike_above_mean(x: Tensor) -> Tensor:
    """Longest consecutive sequence above mean."""
    mu = x.mean(dim=0, keepdim=True)
    mask = x > mu
    return _longest_consecutive_run(mask)


@torch.no_grad()
def longest_strike_below_mean(x: Tensor) -> Tensor:
    """Longest consecutive sequence below mean."""
    mu = x.mean(dim=0, keepdim=True)
    mask = x < mu
    return _longest_consecutive_run(mask)


@torch.no_grad()
def number_crossing_m(x: Tensor, m: float) -> Tensor:
    """Number of crossings of level m."""
    above = x > m
    crossings = (above[1:] != above[:-1]).float().sum(dim=0)
    return crossings


@torch.no_grad()
def number_peaks(x: Tensor, n: int) -> Tensor:
    """Count peaks with support n on each side (vectorized)."""
    length, batch_size, n_states = x.shape

    if 2 * n >= length:
        return torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device)

    # Use max pooling to find local maxima in windows
    # Reshape for pooling: (B*S, 1, N)
    x_reshaped = x.permute(1, 2, 0).reshape(batch_size * n_states, 1, length)

    # Max pool with window size 2n+1 to find max in neighborhood
    window_size = 2 * n + 1
    # Pad to maintain size
    padded = torch.nn.functional.pad(x_reshaped, (n, n), mode="replicate")
    local_max = torch.nn.functional.max_pool1d(padded, kernel_size=window_size, stride=1)

    # A point is a peak if it equals the local max
    is_peak = (x_reshaped == local_max).float()

    # Don't count edges (first n and last n points)
    is_peak[:, :, :n] = 0
    is_peak[:, :, -n:] = 0

    # Sum peaks
    result = is_peak.sum(dim=2).reshape(batch_size, n_states)

    return result


@torch.no_grad()
def number_cwt_peaks(x: Tensor, max_width: int = 5) -> Tensor:
    """Count peaks detected via CWT-like multi-scale analysis (optimized).

    Uses integer accumulation and precomputed masks to minimize allocations
    and avoid unnecessary dtype conversions during the loop.

    Input x: (N, B, S) -> returns (B, S)
    """
    length, batch_size, n_states = x.shape

    valid_widths = [w for w in range(1, max_width + 1) if 2 * w < length]
    if not valid_widths:
        return torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device)

    x_reshaped = x.permute(1, 2, 0).reshape(batch_size * n_states, 1, length)

    total_peaks = torch.zeros(batch_size * n_states, dtype=torch.long, device=x.device)

    for width in valid_widths:
        window_size = 2 * width + 1
        padded = torch.nn.functional.pad(x_reshaped, (width, width), mode="replicate")
        local_max = torch.nn.functional.max_pool1d(padded, kernel_size=window_size, stride=1)

        is_peak = x_reshaped == local_max

        edge_mask = torch.ones(length, dtype=torch.bool, device=x.device)
        edge_mask[:width] = False
        edge_mask[-width:] = False
        is_peak = is_peak & edge_mask.view(1, 1, -1)

        cnt = is_peak.sum(dim=2).squeeze(1).to(torch.long)
        total_peaks += cnt

    avg_peaks = total_peaks.to(x.dtype) / len(valid_widths)
    return avg_peaks.reshape(batch_size, n_states)


# =============================================================================
# BATCHED PATTERN FEATURES
# =============================================================================


@torch.no_grad()
def number_peaks_batched(x: Tensor, ns: list[int]) -> Tensor:
    """Compute number_peaks for multiple n values at once.

    Uses max pooling for each unique n value, computing peaks efficiently.

    Args:
        x: Input tensor of shape (N, B, S)
        ns: List of n values (support on each side)

    Returns:
        Tensor of shape (len(ns), B, S)
    """
    length, batch_size, n_states = x.shape
    x_reshaped = x.permute(1, 2, 0).reshape(batch_size * n_states, 1, length)

    results = []
    for n in ns:
        if 2 * n >= length:
            results.append(torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device))
        else:
            window_size = 2 * n + 1
            padded = torch.nn.functional.pad(x_reshaped, (n, n), mode="replicate")
            local_max = torch.nn.functional.max_pool1d(padded, kernel_size=window_size, stride=1)

            is_peak = (x_reshaped == local_max).float()
            is_peak[:, :, :n] = 0
            is_peak[:, :, -n:] = 0

            result = is_peak.sum(dim=2).reshape(batch_size, n_states)
            results.append(result)

    return torch.stack(results, dim=0)


@torch.no_grad()
def number_crossing_m_batched(x: Tensor, ms: list[float]) -> Tensor:
    """Compute number_crossing_m for multiple m values at once.

    Args:
        x: Input tensor of shape (N, B, S)
        ms: List of threshold values m

    Returns:
        Tensor of shape (len(ms), B, S)
    """
    batch_size, n_states = x.shape[1], x.shape[2]

    if not ms:
        return torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device)

    results = []
    for m in ms:
        above = x > m
        crossings = (above[1:] != above[:-1]).float().sum(dim=0)
        results.append(crossings)

    return torch.stack(results, dim=0)
