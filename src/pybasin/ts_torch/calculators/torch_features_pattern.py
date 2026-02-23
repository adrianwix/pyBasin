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
def find_peak_mask(x: Tensor, n: int = 1) -> Tensor:
    """Find local maxima with strict inequality.

    A point is a peak if it is strictly greater than all neighbors in a window
    of size 2n+1. Edge points (first n and last n) are excluded.

    .. note::
        This matches ``scipy.signal.argrelmax`` behavior (strict inequality),
        not ``scipy.signal.find_peaks`` which also handles flat plateaus by
        returning their middle index. Flat plateaus return no peaks here.

    :param x: Input tensor of shape (N, B, S) where N=time, B=batch, S=states.
    :param n: Support on each side (window size = 2n+1).
    :return: Boolean mask of shape (N, B, S) indicating peak positions.
    """
    length, batch_size, n_states = x.shape

    if 2 * n >= length:
        return torch.zeros(length, batch_size, n_states, dtype=torch.bool, device=x.device)

    window_size = 2 * n + 1
    unfolded = x.unfold(dimension=0, size=window_size, step=1)

    center = unfolded[:, :, :, n]
    left_neighbors = unfolded[:, :, :, :n]
    right_neighbors = unfolded[:, :, :, n + 1 :]

    left_max = left_neighbors.max(dim=-1).values
    right_max = right_neighbors.max(dim=-1).values

    is_peak_interior = (center > left_max) & (center > right_max)

    result = torch.zeros(length, batch_size, n_states, dtype=torch.bool, device=x.device)
    result[n : length - n] = is_peak_interior

    return result


@torch.no_grad()
def number_peaks(x: Tensor, n: int) -> Tensor:
    """Count peaks with support n on each side (vectorized)."""
    mask = find_peak_mask(x, n)
    return mask.float().sum(dim=0)


@torch.no_grad()
def extract_peak_values(x: Tensor, n: int = 1) -> tuple[Tensor, Tensor]:
    """Extract peak amplitude values for orbit diagrams.

    Returns the y-values at detected peaks, useful for visualizing period-N orbits
    where N distinct amplitude levels indicate period-N behavior.

    :param x: Input tensor of shape (N, B, S) where N=time, B=batch, S=states.
    :param n: Support on each side for peak detection (window size = 2n+1).
    :return: Tuple of (peak_values, peak_counts) where:
        - peak_values: Tensor of shape (max_peaks, B, S) with peak amplitudes,
          padded with NaN for trajectories with fewer peaks.
        - peak_counts: Tensor of shape (B, S) with number of peaks per trajectory.
    """
    length, batch_size, n_states = x.shape
    mask = find_peak_mask(x, n)

    peak_counts = mask.sum(dim=0)
    max_peaks = int(peak_counts.max().item()) if peak_counts.numel() > 0 else 0

    if max_peaks == 0:
        return (
            torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device),
            peak_counts,
        )

    peak_values = torch.full(
        (max_peaks, batch_size, n_states), float("nan"), dtype=x.dtype, device=x.device
    )

    indices = mask.nonzero(as_tuple=False)
    if indices.numel() == 0:
        return peak_values, peak_counts

    positions = mask.long().cumsum(dim=0) - 1
    pos_flat = positions[indices[:, 0], indices[:, 1], indices[:, 2]]

    peak_values[pos_flat, indices[:, 1], indices[:, 2]] = x[
        indices[:, 0], indices[:, 1], indices[:, 2]
    ]

    return peak_values, peak_counts


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

    :param x: Input tensor of shape (N, B, S).
    :param ns: List of n values (support on each side).
    :return: Tensor of shape (len(ns), B, S).
    """
    results: list[Tensor] = []
    for n in ns:
        results.append(number_peaks(x, n))
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
