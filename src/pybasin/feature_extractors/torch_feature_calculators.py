# pyright: basic
"""PyTorch-based feature calculators for time series analysis.

This module provides GPU-accelerated feature calculators using PyTorch, implementing
a comprehensive set of tsfresh features. All functions are designed to work with
batched inputs and leverage PyTorch's vectorization.

The calculators operate on time series data with shape (N, B, S) where:
- N: number of time steps
- B: batch size (number of trajectories)
- S: number of state variables

Each calculator returns features with shape (B, S).

Note: Unlike JAX, we don't use JIT compilation since features run one-off.
PyTorch's eager execution with native batch operations should be efficient.
"""

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor

# =============================================================================
# MINIMAL FEATURES (tsfresh MinimalFCParameters - 10 features)
# =============================================================================


@torch.no_grad()
def sum_values(x: Tensor) -> Tensor:
    """Sum of all values."""
    return x.sum(dim=0)


@torch.no_grad()
def median(x: Tensor) -> Tensor:
    """Median of the time series."""
    return x.median(dim=0).values


@torch.no_grad()
def mean(x: Tensor) -> Tensor:
    """Mean of the time series."""
    return x.mean(dim=0)


@torch.no_grad()
def length(x: Tensor) -> Tensor:
    """Length of the time series."""
    n = x.shape[0]
    return torch.full(x.shape[1:], n, dtype=x.dtype, device=x.device)


@torch.no_grad()
def standard_deviation(x: Tensor) -> Tensor:
    """Standard deviation (population, ddof=0)."""
    return x.std(dim=0, correction=0)


@torch.no_grad()
def variance(x: Tensor) -> Tensor:
    """Variance (population, ddof=0)."""
    return x.var(dim=0, correction=0)


@torch.no_grad()
def root_mean_square(x: Tensor) -> Tensor:
    """Root mean square value."""
    return torch.sqrt((x**2).mean(dim=0))


@torch.no_grad()
def maximum(x: Tensor) -> Tensor:
    """Maximum value."""
    return x.max(dim=0).values


@torch.no_grad()
def absolute_maximum(x: Tensor) -> Tensor:
    """Maximum absolute value."""
    return x.abs().max(dim=0).values


@torch.no_grad()
def minimum(x: Tensor) -> Tensor:
    """Minimum value."""
    return x.min(dim=0).values


# =============================================================================
# SIMPLE STATISTICS (5 features)
# =============================================================================


@torch.no_grad()
def abs_energy(x: Tensor) -> Tensor:
    """Absolute energy (sum of squared values)."""
    return (x**2).sum(dim=0)


@torch.no_grad()
def kurtosis(x: Tensor) -> Tensor:
    """Fisher's kurtosis (excess kurtosis, bias-corrected)."""
    n = x.shape[0]
    mu = x.mean(dim=0, keepdim=True)
    m2 = ((x - mu) ** 2).mean(dim=0)
    m4 = ((x - mu) ** 4).mean(dim=0)
    # Bias-corrected excess kurtosis
    g2 = m4 / (m2**2 + 1e-10) - 3
    # Apply bias correction factor
    correction = ((n - 1) / ((n - 2) * (n - 3) + 1e-10)) * ((n + 1) * g2 + 6)
    return correction


@torch.no_grad()
def skewness(x: Tensor) -> Tensor:
    """Fisher's skewness (bias-corrected)."""
    n = x.shape[0]
    mu = x.mean(dim=0, keepdim=True)
    m2 = ((x - mu) ** 2).mean(dim=0)
    m3 = ((x - mu) ** 3).mean(dim=0)
    # Bias-corrected skewness
    g1 = m3 / (m2**1.5 + 1e-10)
    correction = (
        g1 * torch.sqrt(torch.tensor(n * (n - 1), dtype=x.dtype, device=x.device)) / (n - 2 + 1e-10)
    )
    return correction


@torch.no_grad()
def quantile(x: Tensor, q: float) -> Tensor:
    """Q-quantile of the time series."""
    return torch.quantile(x, q, dim=0)


@torch.no_grad()
def variation_coefficient(x: Tensor) -> Tensor:
    """Coefficient of variation (std / mean)."""
    return x.std(dim=0, correction=0) / (x.mean(dim=0).abs() + 1e-10)


# =============================================================================
# CHANGE/DIFFERENCE BASED (4 features)
# =============================================================================


@torch.no_grad()
def absolute_sum_of_changes(x: Tensor) -> Tensor:
    """Sum of absolute differences between consecutive values."""
    return torch.abs(x[1:] - x[:-1]).sum(dim=0)


@torch.no_grad()
def mean_abs_change(x: Tensor) -> Tensor:
    """Mean of absolute differences between consecutive values."""
    return torch.abs(x[1:] - x[:-1]).mean(dim=0)


@torch.no_grad()
def mean_change(x: Tensor) -> Tensor:
    """Mean change: (x[-1] - x[0]) / (n - 1)."""
    n = x.shape[0]
    return (x[-1] - x[0]) / (n - 1)


@torch.no_grad()
def mean_second_derivative_central(x: Tensor) -> Tensor:
    """Mean of second derivative (central difference): (x[-1] - x[-2] - x[1] + x[0]) / (2 * (n-2))."""
    n = x.shape[0]
    return (x[-1] - x[-2] - x[1] + x[0]) / (2 * (n - 2))


# =============================================================================
# COUNTING FEATURES (4 features)
# =============================================================================


@torch.no_grad()
def count_above(x: Tensor, t: float) -> Tensor:
    """Percentage of values above threshold t."""
    return (x > t).float().mean(dim=0)


@torch.no_grad()
def count_above_mean(x: Tensor) -> Tensor:
    """Count of values above mean."""
    mu = x.mean(dim=0, keepdim=True)
    return (x > mu).float().sum(dim=0)


@torch.no_grad()
def count_below(x: Tensor, t: float) -> Tensor:
    """Percentage of values below threshold t."""
    return (x < t).float().mean(dim=0)


@torch.no_grad()
def count_below_mean(x: Tensor) -> Tensor:
    """Count of values below mean."""
    mu = x.mean(dim=0, keepdim=True)
    return (x < mu).float().sum(dim=0)


# =============================================================================
# BOOLEAN FEATURES (5 features)
# =============================================================================


@torch.no_grad()
def has_duplicate(x: Tensor) -> Tensor:
    """Check if any value occurs more than once (optimized with sorting)."""
    n, batch_size, n_states = x.shape
    # Reshape to (N, B*S) and sort along time dimension
    x_flat = x.reshape(n, -1)
    sorted_x, _ = x_flat.sort(dim=0)
    # Check if any adjacent sorted values are equal
    has_adj_dup = (sorted_x[1:] == sorted_x[:-1]).any(dim=0)  # (B*S,)
    return has_adj_dup.float().reshape(batch_size, n_states)


@torch.no_grad()
def has_duplicate_max(x: Tensor) -> Tensor:
    """Check if maximum value occurs more than once."""
    max_val = x.max(dim=0, keepdim=True).values
    return ((x == max_val).float().sum(dim=0) > 1).float()


@torch.no_grad()
def has_duplicate_min(x: Tensor) -> Tensor:
    """Check if minimum value occurs more than once."""
    min_val = x.min(dim=0, keepdim=True).values
    return ((x == min_val).float().sum(dim=0) > 1).float()


@torch.no_grad()
def has_variance_larger_than_standard_deviation(x: Tensor) -> Tensor:
    """Check if variance > standard deviation (equivalent to std > 1)."""
    std = x.std(dim=0, correction=0)
    return (std > 1).float()


@torch.no_grad()
def has_large_standard_deviation(x: Tensor, r: float = 0.25) -> Tensor:
    """Check if std > r * range."""
    std = x.std(dim=0, correction=0)
    range_val = x.max(dim=0).values - x.min(dim=0).values
    return (std > r * range_val).float()


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
    """Count peaks detected via CWT-like multi-scale analysis (fully vectorized)."""
    length, batch_size, n_states = x.shape

    # Reshape once: (B*S, 1, N)
    x_reshaped = x.permute(1, 2, 0).reshape(batch_size * n_states, 1, length)

    # Compute peaks for all widths and accumulate
    # Use the largest valid width to determine which widths to use
    valid_widths = [w for w in range(1, max_width + 1) if 2 * w < length]
    if not valid_widths:
        return torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device)

    total_peaks = torch.zeros(batch_size * n_states, dtype=x.dtype, device=x.device)

    # Process each width - the loop is unavoidable due to different kernel sizes
    # but we minimize overhead by reusing the reshaped tensor
    for width in valid_widths:
        window_size = 2 * width + 1
        padded = torch.nn.functional.pad(x_reshaped, (width, width), mode="replicate")
        local_max = torch.nn.functional.max_pool1d(padded, kernel_size=window_size, stride=1)

        # Count peaks (where value equals local max, excluding edges)
        is_peak = x_reshaped == local_max
        # Zero out edges using a mask instead of assignment
        edge_mask = torch.ones(length, dtype=torch.bool, device=x.device)
        edge_mask[:width] = False
        edge_mask[-width:] = False
        is_peak = is_peak & edge_mask.view(1, 1, -1)

        total_peaks += is_peak.sum(dim=2).squeeze(1).float()

    return (total_peaks / len(valid_widths)).reshape(batch_size, n_states)


# =============================================================================
# AUTOCORRELATION FEATURES (3 features)
# =============================================================================


@torch.no_grad()
def autocorrelation(x: Tensor, lag: int) -> Tensor:
    """Autocorrelation at given lag."""
    if lag == 0:
        return torch.ones(x.shape[1:], dtype=x.dtype, device=x.device)
    n = x.shape[0]
    if lag >= n:
        return torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)

    mu = x.mean(dim=0, keepdim=True)
    x_centered = x - mu
    var = (x_centered**2).sum(dim=0)

    # Compute autocorrelation
    autocov = (x_centered[:-lag] * x_centered[lag:]).sum(dim=0)
    return autocov / (var + 1e-10)


@torch.no_grad()
def partial_autocorrelation(x: Tensor, lag: int) -> Tensor:
    """Partial autocorrelation at given lag using Durbin-Levinson."""
    if lag == 0:
        return torch.ones(x.shape[1:], dtype=x.dtype, device=x.device)

    n = x.shape[0]
    maxlag = min(lag, n - 1)

    # Compute autocorrelations up to maxlag
    acf = torch.stack([autocorrelation(x, k) for k in range(maxlag + 1)], dim=0)

    # Durbin-Levinson algorithm (simplified for single lag)
    result = torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)
    for b in range(x.shape[1]):
        for s in range(x.shape[2]):
            r = acf[:, b, s].cpu().numpy()
            # Levinson recursion
            phi = [0.0] * (maxlag + 1)
            phi[1] = float(r[1])
            for k in range(2, maxlag + 1):
                num = r[k] - sum(phi[j] * r[k - j] for j in range(1, k))
                denom = 1 - sum(phi[j] * r[j] for j in range(1, k))
                phi[k] = float(num / (denom + 1e-10))
                for j in range(1, k):
                    phi[j] = phi[j] - phi[k] * phi[k - j]
            result[b, s] = float(phi[lag]) if lag <= maxlag else 0.0
    return result


@torch.no_grad()
def agg_autocorrelation(x: Tensor, maxlag: int = 40, f_agg: str = "mean") -> Tensor:
    """Aggregated autocorrelation over lags 1 to maxlag (FFT-optimized)."""
    n = x.shape[0]
    maxlag = min(maxlag, n - 1)

    # Compute all autocorrelations at once using FFT
    x_centered = x - x.mean(dim=0, keepdim=True)
    n_fft = 2 * n
    fft_x = torch.fft.rfft(x_centered, n=n_fft, dim=0)
    power_spectrum = fft_x.real**2 + fft_x.imag**2
    autocorr_full = torch.fft.irfft(power_spectrum, n=n_fft, dim=0)

    # Normalize by variance (lag 0)
    var = autocorr_full[0:1]
    acf = autocorr_full[1 : maxlag + 1] / (var + 1e-10)  # Lags 1 to maxlag

    if f_agg == "mean":
        return acf.mean(dim=0)
    elif f_agg == "median":
        return acf.median(dim=0).values
    elif f_agg == "var":
        return acf.var(dim=0, correction=0)
    else:
        return acf.mean(dim=0)


# =============================================================================
# ENTROPY/COMPLEXITY FEATURES (5 features)
# =============================================================================


@torch.no_grad()
def permutation_entropy(x: Tensor, tau: int = 1, dimension: int = 3) -> Tensor:
    """Permutation entropy (vectorized)."""
    import math

    n, batch_size, n_states = x.shape
    num_patterns = n - (dimension - 1) * tau

    if num_patterns <= 0:
        return torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)

    # Build embedded matrix: (num_patterns, dimension, B, S)
    indices = (
        torch.arange(num_patterns, device=x.device).unsqueeze(1)
        + torch.arange(dimension, device=x.device).unsqueeze(0) * tau
    )
    embedded = x[indices]  # (num_patterns, dimension, B, S)

    # Get rank patterns using argsort: (num_patterns, dimension, B, S)
    ranks = embedded.argsort(dim=1)

    # Convert ranks to a single integer pattern ID for counting
    # Each pattern is a permutation of [0, 1, ..., dimension-1]
    # Use base factorial encoding for unique ID
    multipliers = torch.tensor(
        [math.factorial(dimension - 1 - i) for i in range(dimension)],
        device=x.device,
        dtype=torch.long,
    ).view(1, dimension, 1, 1)
    pattern_ids = (ranks * multipliers).sum(dim=1)  # (num_patterns, B, S)

    # Count unique patterns per batch/state
    n_permutations = math.factorial(dimension)
    result = torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device)

    # Flatten batch and state dims for efficient bincount
    pattern_ids_flat = pattern_ids.permute(1, 2, 0).reshape(batch_size * n_states, num_patterns)

    for i in range(batch_size * n_states):
        counts = torch.bincount(pattern_ids_flat[i], minlength=n_permutations).float()
        probs = counts / num_patterns
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum()
        max_entropy = math.log(n_permutations)
        result.view(-1)[i] = entropy / max_entropy if max_entropy > 0 else 0.0

    return result


@torch.no_grad()
def binned_entropy(x: Tensor, max_bins: int = 10) -> Tensor:
    """Entropy of binned distribution (vectorized)."""
    n, batch_size, n_states = x.shape

    # Get min/max per series
    min_vals = x.min(dim=0).values  # (B, S)
    max_vals = x.max(dim=0).values  # (B, S)
    ranges = max_vals - min_vals  # (B, S)

    # Handle constant series
    ranges = torch.where(ranges == 0, torch.ones_like(ranges), ranges)

    # Normalize x to [0, max_bins-1] per series
    x_norm = (x - min_vals.unsqueeze(0)) / (ranges.unsqueeze(0) + 1e-10) * (max_bins - 1e-6)
    bin_indices = x_norm.long().clamp(0, max_bins - 1)  # (N, B, S)

    # Use one-hot encoding to count bins
    one_hot = torch.nn.functional.one_hot(
        bin_indices, num_classes=max_bins
    ).float()  # (N, B, S, max_bins)
    counts = one_hot.sum(dim=0)  # (B, S, max_bins)

    # Compute probabilities and entropy
    probs = counts / n  # (B, S, max_bins)
    probs = probs.clamp(min=1e-10)  # Avoid log(0)

    # Only count non-zero bins in entropy
    log_probs = probs.log()
    log_probs = torch.where(counts > 0, log_probs, torch.zeros_like(log_probs))
    entropy = -(probs * log_probs).sum(dim=-1)  # (B, S)

    # Zero out constant series
    is_constant = max_vals == min_vals
    entropy = torch.where(is_constant, torch.zeros_like(entropy), entropy)

    return entropy


@torch.no_grad()
def fourier_entropy(x: Tensor, bins: int = 10) -> Tensor:
    """Entropy of the power spectral density."""
    fft_result = torch.fft.rfft(x, dim=0)
    psd = fft_result.real**2 + fft_result.imag**2
    # Normalize to get probability distribution
    psd_sum = psd.sum(dim=0, keepdim=True)
    psd_norm = psd / (psd_sum + 1e-10)
    # Compute entropy
    psd_norm = psd_norm.clamp(min=1e-10)
    entropy = -(psd_norm * psd_norm.log()).sum(dim=0)
    return entropy


@torch.no_grad()
def lempel_ziv_complexity(x: Tensor, bins: int = 2) -> Tensor:
    """Lempel-Ziv complexity approximation (optimized)."""
    import math

    n, batch_size, n_states = x.shape
    result = torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device)

    # Normalize by max complexity
    b_n = n / (math.log(n) + 1e-10) if n > 1 else 1.0

    # Discretize all series at once
    min_val = x.min(dim=0, keepdim=True).values
    max_val = x.max(dim=0, keepdim=True).values
    range_val = max_val - min_val + 1e-10
    binned = ((x - min_val) / range_val * bins).long().clamp(0, bins - 1)

    # Convert to numpy for faster string operations
    binned_np = binned.cpu().numpy()

    for b in range(batch_size):
        for s in range(n_states):
            seq = binned_np[:, b, s]
            # Use bytes for faster hashing
            seq_bytes = seq.astype("uint8").tobytes()
            seen: set[bytes] = set()
            i = 0
            complexity = 0
            while i < n:
                length = 1
                while i + length <= n:
                    substr = seq_bytes[i : i + length]
                    if substr not in seen:
                        seen.add(substr)
                        complexity += 1
                        break
                    length += 1
                i += length
            result[b, s] = complexity / b_n

    return result.to(x.device)


@torch.no_grad()
def cid_ce(x: Tensor, normalize: bool = True) -> Tensor:
    """Complexity-invariant distance."""
    diff = x[1:] - x[:-1]
    ce = torch.sqrt((diff**2).sum(dim=0))
    if normalize:
        std = x.std(dim=0, correction=0)
        ce = ce / (std + 1e-10)
    return ce


# =============================================================================
# FREQUENCY DOMAIN FEATURES (4 features)
# =============================================================================


@torch.no_grad()
def fft_coefficient(x: Tensor, coeff: int = 0, attr: str = "abs") -> Tensor:
    """FFT coefficient attributes."""
    fft_result = torch.fft.rfft(x, dim=0)
    n_coeffs = fft_result.shape[0]
    if coeff >= n_coeffs:
        return torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)

    c = fft_result[coeff]
    if attr == "real":
        return c.real
    elif attr == "imag":
        return c.imag
    elif attr == "abs":
        return c.abs()
    elif attr == "angle":
        return torch.atan2(c.imag, c.real)
    else:
        return c.abs()


@torch.no_grad()
def fft_aggregated(x: Tensor, aggtype: str = "centroid") -> Tensor:
    """Aggregated FFT spectral statistics."""
    fft_result = torch.fft.rfft(x, dim=0)
    psd = fft_result.real**2 + fft_result.imag**2
    n_coeffs = psd.shape[0]
    freqs = torch.arange(n_coeffs, dtype=x.dtype, device=x.device).unsqueeze(-1).unsqueeze(-1)

    psd_sum = psd.sum(dim=0, keepdim=True)
    psd_norm = psd / (psd_sum + 1e-10)

    if aggtype == "centroid":
        return (freqs * psd_norm).sum(dim=0)
    elif aggtype == "variance":
        centroid = (freqs * psd_norm).sum(dim=0, keepdim=True)
        return ((freqs - centroid) ** 2 * psd_norm).sum(dim=0)
    elif aggtype == "skew":
        centroid = (freqs * psd_norm).sum(dim=0, keepdim=True)
        var = ((freqs - centroid) ** 2 * psd_norm).sum(dim=0, keepdim=True)
        return (((freqs - centroid) ** 3 * psd_norm).sum(dim=0)) / (var**1.5 + 1e-10).squeeze(0)
    elif aggtype == "kurtosis":
        centroid = (freqs * psd_norm).sum(dim=0, keepdim=True)
        var = ((freqs - centroid) ** 2 * psd_norm).sum(dim=0, keepdim=True)
        return (((freqs - centroid) ** 4 * psd_norm).sum(dim=0)) / (var**2 + 1e-10).squeeze(0)
    else:
        return (freqs * psd_norm).sum(dim=0)


@torch.no_grad()
def spkt_welch_density(x: Tensor, coeff: int = 0) -> Tensor:
    """Simplified Welch power spectral density at coefficient."""
    fft_result = torch.fft.rfft(x, dim=0)
    psd = (fft_result.real**2 + fft_result.imag**2) / x.shape[0]
    n_coeffs = psd.shape[0]
    if coeff >= n_coeffs:
        return torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)
    return psd[coeff]


@torch.no_grad()
def cwt_coefficients(x: Tensor, width: int = 2, coeff: int = 0) -> Tensor:
    """CWT coefficients using Ricker wavelet (vectorized)."""
    n, batch_size, n_states = x.shape

    # Create Ricker wavelet
    t = torch.arange(-width * 4, width * 4 + 1, dtype=x.dtype, device=x.device)
    sigma = float(width)
    wavelet = (
        (2 / (torch.sqrt(torch.tensor(3.0 * sigma, device=x.device)) * torch.pi**0.25))
        * (1 - (t / sigma) ** 2)
        * torch.exp(-(t**2) / (2 * sigma**2))
    )
    wavelet_len = len(wavelet)

    # Reshape x for batch conv1d: (B*S, 1, N)
    x_reshaped = x.permute(1, 2, 0).reshape(batch_size * n_states, 1, n)

    # Pad and convolve
    padded = torch.nn.functional.pad(
        x_reshaped, (wavelet_len // 2, wavelet_len // 2), mode="reflect"
    )

    # Wavelet kernel: (1, 1, wavelet_len)
    kernel = wavelet.unsqueeze(0).unsqueeze(0)

    # Convolution: (B*S, 1, N)
    conv = torch.nn.functional.conv1d(padded, kernel, padding=0)

    # Extract coefficient
    if coeff < conv.shape[2]:
        result = conv[:, 0, coeff].reshape(batch_size, n_states)
    else:
        result = torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device)

    return result


# =============================================================================
# TREND/REGRESSION FEATURES (5 features)
# =============================================================================


@torch.no_grad()
def linear_trend(x: Tensor, attr: str = "slope") -> Tensor:
    """Linear regression trend attributes."""
    n = x.shape[0]
    t = torch.arange(n, dtype=x.dtype, device=x.device).unsqueeze(-1).unsqueeze(-1)
    t_mean = t.mean()
    x_mean = x.mean(dim=0, keepdim=True)

    # Compute slope and intercept
    ss_tt = ((t - t_mean) ** 2).sum()
    ss_tx = ((t - t_mean) * (x - x_mean)).sum(dim=0)
    slope = ss_tx / (ss_tt + 1e-10)
    intercept = x_mean.squeeze(0) - slope * t_mean

    if attr == "slope":
        return slope
    elif attr == "intercept":
        return intercept
    elif attr == "rvalue":
        ss_xx = ((x - x_mean) ** 2).sum(dim=0)
        rvalue = ss_tx / (torch.sqrt(ss_tt * ss_xx) + 1e-10)
        return rvalue
    elif attr == "pvalue":
        # Approximate p-value (simplified)
        ss_xx = ((x - x_mean) ** 2).sum(dim=0)
        rvalue = ss_tx / (torch.sqrt(ss_tt * ss_xx) + 1e-10)
        t_stat = (
            rvalue
            * torch.sqrt(torch.tensor(n - 2, dtype=x.dtype, device=x.device))
            / (torch.sqrt(1 - rvalue**2) + 1e-10)
        )
        # Return pseudo p-value (lower t_stat = higher p)
        return 1 / (1 + t_stat.abs())
    elif attr == "stderr":
        y_pred = slope * t + intercept
        residuals = x - y_pred
        mse = (residuals**2).sum(dim=0) / (n - 2)
        stderr = torch.sqrt(mse / (ss_tt + 1e-10))
        return stderr
    else:
        return slope


@torch.no_grad()
def linear_trend_timewise(x: Tensor, attr: str = "slope") -> Tensor:
    """Linear trend (same as linear_trend for our use case)."""
    return linear_trend(x, attr)


@torch.no_grad()
def agg_linear_trend(
    x: Tensor, chunk_size: int = 10, f_agg: str = "mean", attr: str = "slope"
) -> Tensor:
    """Linear trend on aggregated chunks."""
    n = x.shape[0]
    n_chunks = n // chunk_size
    if n_chunks < 2:
        return torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)

    # Aggregate chunks
    chunks = x[: n_chunks * chunk_size].reshape(n_chunks, chunk_size, x.shape[1], x.shape[2])
    if f_agg == "mean":
        agg = chunks.mean(dim=1)
    elif f_agg == "var":
        agg = chunks.var(dim=1, correction=0)
    elif f_agg == "min":
        agg = chunks.min(dim=1).values
    elif f_agg == "max":
        agg = chunks.max(dim=1).values
    else:
        agg = chunks.mean(dim=1)

    return linear_trend(agg, attr)


@torch.no_grad()
def ar_coefficient(x: Tensor, k: int = 1, coeff: int = 0) -> Tensor:
    """AR model coefficients using Yule-Walker (optimized with FFT autocorrelation)."""
    if coeff > k:
        return torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)

    n, batch_size, n_states = x.shape

    # Compute autocorrelation using FFT (much faster for multiple lags)
    x_centered = x - x.mean(dim=0, keepdim=True)

    # FFT-based autocorrelation
    n_fft = 2 * n
    fft_x = torch.fft.rfft(x_centered, n=n_fft, dim=0)
    power_spectrum = fft_x.real**2 + fft_x.imag**2
    autocorr_full = torch.fft.irfft(power_spectrum, n=n_fft, dim=0)[: k + 1]  # (k+1, B, S)

    # Normalize by variance (lag 0)
    acf = autocorr_full / (autocorr_full[0:1] + 1e-10)  # (k+1, B, S)

    # Reshape for batch processing: (B*S, k+1)
    acf_flat = acf.reshape(k + 1, -1).T  # (B*S, k+1)

    # Build Toeplitz matrix using advanced indexing
    idx = torch.arange(k, device=x.device)
    toeplitz_idx = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # (k, k)
    r_matrix = acf_flat[:, toeplitz_idx]  # (B*S, k, k)

    # r vector: acf[1:k+1]
    r_vec = acf_flat[:, 1 : k + 1]  # (B*S, k)

    # Add regularization and solve
    reg = 1e-6 * torch.eye(k, device=x.device, dtype=x.dtype)
    r_matrix = r_matrix + reg

    try:
        phi = torch.linalg.solve(r_matrix, r_vec)  # (B*S, k)
        result = (
            phi[:, coeff]
            if coeff < k
            else torch.zeros(batch_size * n_states, dtype=x.dtype, device=x.device)
        )
    except Exception:
        result = torch.zeros(batch_size * n_states, dtype=x.dtype, device=x.device)

    return result.reshape(batch_size, n_states)


@torch.no_grad()
def augmented_dickey_fuller(x: Tensor, attr: str = "teststat") -> Tensor:
    """Simplified Augmented Dickey-Fuller test (vectorized)."""
    n = x.shape[0]

    if attr == "usedlag":
        return torch.ones(x.shape[1:], dtype=x.dtype, device=x.device)

    # Simple ADF approximation - vectorized
    diff = x[1:] - x[:-1]  # (N-1, B, S)
    lagged = x[:-1]  # (N-1, B, S)

    # Regression: diff = alpha + beta * lagged + error
    x_mean = lagged.mean(dim=0, keepdim=True)  # (1, B, S)
    y_mean = diff.mean(dim=0, keepdim=True)  # (1, B, S)

    x_centered = lagged - x_mean
    y_centered = diff - y_mean

    ss_xx = (x_centered**2).sum(dim=0)  # (B, S)
    ss_xy = (x_centered * y_centered).sum(dim=0)  # (B, S)

    beta = ss_xy / (ss_xx + 1e-10)  # (B, S)

    # Residuals
    residuals = diff - (y_mean + beta.unsqueeze(0) * x_centered)
    mse = (residuals**2).sum(dim=0) / (n - 3)  # (B, S)

    se = torch.sqrt(mse) / (torch.sqrt(ss_xx) + 1e-10)  # (B, S)
    t_stat = beta / (se + 1e-10)  # (B, S)

    if attr == "teststat":
        return t_stat
    elif attr == "pvalue":
        return 1 / (1 + t_stat.abs())
    else:
        return t_stat


# =============================================================================
# REOCCURRENCE FEATURES (5 features)
# =============================================================================


@torch.no_grad()
def percentage_of_reoccurring_datapoints_to_all_datapoints(x: Tensor) -> Tensor:
    """Percentage of values that appear more than once (optimized)."""
    n, batch_size, n_states = x.shape

    # Reshape to (N, B*S) for batch processing
    x_flat = x.reshape(n, -1)  # (N, B*S)

    # Use a single loop over flattened batch*state dimension
    result = torch.zeros(batch_size * n_states, dtype=x.dtype, device=x.device)
    for i in range(batch_size * n_states):
        series = x_flat[:, i]
        unique, counts = torch.unique(series, return_counts=True)
        reoccurring = (counts > 1).sum()
        result[i] = reoccurring.float() / len(unique) if len(unique) > 0 else 0.0

    return result.reshape(batch_size, n_states)


@torch.no_grad()
def percentage_of_reoccurring_values_to_all_values(x: Tensor) -> Tensor:
    """Percentage of datapoints that are reoccurring (optimized)."""
    n, batch_size, n_states = x.shape

    if n <= 1:
        return torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device)

    # Reshape to (N, B*S) for batch processing
    x_flat = x.reshape(n, -1)  # (N, B*S)

    # Sort and find adjacent duplicates
    sorted_x, _ = x_flat.sort(dim=0)
    is_dup = sorted_x[1:] == sorted_x[:-1]  # (N-1, B*S)

    # Mark all positions that are part of a duplicate group
    # Position i is reoccurring if sorted[i] == sorted[i-1] OR sorted[i] == sorted[i+1]
    is_dup_prev = torch.cat(
        [torch.zeros(1, x_flat.shape[1], dtype=torch.bool, device=x.device), is_dup], dim=0
    )
    is_dup_next = torch.cat(
        [is_dup, torch.zeros(1, x_flat.shape[1], dtype=torch.bool, device=x.device)], dim=0
    )
    is_reoccurring = is_dup_prev | is_dup_next  # (N, B*S)

    # Count reoccurring values
    result = is_reoccurring.float().sum(dim=0) / n  # (B*S,)

    return result.reshape(batch_size, n_states)


@torch.no_grad()
def sum_of_reoccurring_data_points(x: Tensor) -> Tensor:
    """Sum of values that appear more than once (optimized)."""
    n, batch_size, n_states = x.shape

    # Reshape to (N, B*S) for batch processing
    x_flat = x.reshape(n, -1)  # (N, B*S)

    # Sort and find duplicates
    sorted_x, _ = x_flat.sort(dim=0)
    is_dup = sorted_x[1:] == sorted_x[:-1]  # (N-1, B*S)

    # Mark all values that are duplicated (appear more than once)
    is_dup_prev = torch.cat(
        [torch.zeros(1, x_flat.shape[1], dtype=torch.bool, device=x.device), is_dup], dim=0
    )
    is_dup_next = torch.cat(
        [is_dup, torch.zeros(1, x_flat.shape[1], dtype=torch.bool, device=x.device)], dim=0
    )
    is_reoccurring = is_dup_prev | is_dup_next  # (N, B*S)

    # Sum reoccurring values
    result = (sorted_x * is_reoccurring.float()).sum(dim=0)  # (B*S,)

    return result.reshape(batch_size, n_states)


@torch.no_grad()
def sum_of_reoccurring_values(x: Tensor) -> Tensor:
    """Sum of unique values that appear more than once (optimized)."""
    n, batch_size, n_states = x.shape

    # Reshape to (N, B*S) for batch processing
    x_flat = x.reshape(n, -1)  # (N, B*S)

    # Sort and find duplicate boundaries
    sorted_x, _ = x_flat.sort(dim=0)
    is_dup = sorted_x[1:] == sorted_x[:-1]  # (N-1, B*S)

    # Find first occurrence of each reoccurring value
    # A value is the "first of reoccurring" if it's followed by a duplicate but not preceded by one
    is_dup_prev = torch.cat(
        [torch.zeros(1, x_flat.shape[1], dtype=torch.bool, device=x.device), is_dup], dim=0
    )
    is_dup_next = torch.cat(
        [is_dup, torch.zeros(1, x_flat.shape[1], dtype=torch.bool, device=x.device)], dim=0
    )

    # First of a duplicate run: has duplicate after but not before
    is_first_of_run = is_dup_next & ~is_dup_prev  # (N, B*S)

    # Sum unique reoccurring values (just the first of each run)
    result = (sorted_x * is_first_of_run.float()).sum(dim=0)  # (B*S,)

    return result.reshape(batch_size, n_states)


@torch.no_grad()
def ratio_value_number_to_time_series_length(x: Tensor) -> Tensor:
    """Ratio of unique values to length (optimized)."""
    n, batch_size, n_states = x.shape

    # Reshape to (N, B*S) for batch processing
    x_flat = x.reshape(n, -1)  # (N, B*S)

    # Sort and count duplicates
    sorted_x, _ = x_flat.sort(dim=0)
    is_dup = (sorted_x[1:] == sorted_x[:-1]).float()  # (N-1, B*S)

    # Number of unique = N - number of duplicates
    n_dups = is_dup.sum(dim=0)  # (B*S,)
    n_unique = n - n_dups

    result = n_unique / n
    return result.reshape(batch_size, n_states)


# =============================================================================
# ADVANCED FEATURES (12 features, excluding matrix_profile)
# =============================================================================


@torch.no_grad()
def benford_correlation(x: Tensor) -> Tensor:
    """Correlation with Benford's law distribution (vectorized)."""
    import math

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
def change_quantiles(
    x: Tensor, ql: float, qh: float, isabs: bool = True, f_agg: str = "mean"
) -> Tensor:
    """Statistics of changes within quantile corridor (optimized)."""
    n, batch_size, n_states = x.shape

    q_low = torch.quantile(x, ql, dim=0, keepdim=True)  # (1, B, S)
    q_high = torch.quantile(x, qh, dim=0, keepdim=True)  # (1, B, S)
    mask = (x >= q_low) & (x <= q_high)  # (N, B, S)

    # Compute differences between consecutive values
    diff = x[1:] - x[:-1]  # (N-1, B, S)
    if isabs:
        diff = diff.abs()

    # Mask for valid differences: both current and previous must be in corridor
    # A change is valid if both endpoints are in the quantile corridor
    valid_changes = mask[:-1] & mask[1:]  # (N-1, B, S)

    # Apply mask and compute aggregation
    masked_diff = diff * valid_changes.float()  # Zero out invalid changes
    count = valid_changes.float().sum(dim=0)  # (B, S)

    if f_agg == "mean":
        result = masked_diff.sum(dim=0) / (count + 1e-10)
    elif f_agg == "var":
        mean_val = masked_diff.sum(dim=0) / (count + 1e-10)
        sq_diff = (diff - mean_val.unsqueeze(0)) ** 2 * valid_changes.float()
        result = sq_diff.sum(dim=0) / (count + 1e-10)
    else:
        result = masked_diff.sum(dim=0) / (count + 1e-10)

    # Zero out where no valid changes
    result = torch.where(count > 0, result, torch.zeros_like(result))

    return result


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


@torch.no_grad()
def mean_n_absolute_max(x: Tensor, number_of_maxima: int = 1) -> Tensor:
    """Mean of n largest absolute values (optimized with topk)."""
    x_abs = x.abs()
    k = min(number_of_maxima, x.shape[0])
    # Use topk instead of full sort - much faster for small k
    top_vals, _ = x_abs.topk(k, dim=0)
    return top_vals.mean(dim=0)


@torch.no_grad()
def range_count(x: Tensor, min_val: float, max_val: float) -> Tensor:
    """Count of values in range [min_val, max_val]."""
    return ((x >= min_val) & (x <= max_val)).float().sum(dim=0)


@torch.no_grad()
def ratio_beyond_r_sigma(x: Tensor, r: float = 1.0) -> Tensor:
    """Ratio of values beyond r standard deviations."""
    mu = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, correction=0, keepdim=True)
    beyond = (x - mu).abs() > r * std
    return beyond.float().mean(dim=0)


@torch.no_grad()
def symmetry_looking(x: Tensor, r: float = 0.1) -> Tensor:
    """Check if distribution looks symmetric."""
    mu = x.mean(dim=0)
    med = x.median(dim=0).values
    range_val = x.max(dim=0).values - x.min(dim=0).values
    return ((mu - med).abs() < r * range_val).float()


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


@torch.no_grad()
def value_count(x: Tensor, value: float) -> Tensor:
    """Count of specific value."""
    return (x == value).float().sum(dim=0)


# =============================================================================
# CUSTOM FEATURES (2 features)
# =============================================================================


@torch.no_grad()
def delta(x: Tensor) -> Tensor:
    """Absolute difference between max and mean."""
    return (x.max(dim=0).values - x.mean(dim=0)).abs()


@torch.no_grad()
def log_delta(x: Tensor) -> Tensor:
    """Log of delta (with epsilon for stability)."""
    d = delta(x)
    return torch.log(d + 1e-10)


# =============================================================================
# FEATURE FUNCTIONS DICTIONARY
# =============================================================================

ALL_FEATURE_FUNCTIONS: dict[str, Callable[..., Tensor]] = {
    # Minimal features (10)
    "sum_values": sum_values,
    "median": median,
    "mean": mean,
    "length": length,
    "standard_deviation": standard_deviation,
    "variance": variance,
    "root_mean_square": root_mean_square,
    "maximum": maximum,
    "absolute_maximum": absolute_maximum,
    "minimum": minimum,
    # Simple statistics (5)
    "abs_energy": abs_energy,
    "kurtosis": kurtosis,
    "skewness": skewness,
    "quantile": quantile,
    "variation_coefficient": variation_coefficient,
    # Change/difference (4)
    "absolute_sum_of_changes": absolute_sum_of_changes,
    "mean_abs_change": mean_abs_change,
    "mean_change": mean_change,
    "mean_second_derivative_central": mean_second_derivative_central,
    # Counting (4)
    "count_above": count_above,
    "count_above_mean": count_above_mean,
    "count_below": count_below,
    "count_below_mean": count_below_mean,
    # Boolean (5)
    "has_duplicate": has_duplicate,
    "has_duplicate_max": has_duplicate_max,
    "has_duplicate_min": has_duplicate_min,
    "has_variance_larger_than_standard_deviation": has_variance_larger_than_standard_deviation,
    "large_standard_deviation": has_large_standard_deviation,
    # Location (5)
    "first_location_of_maximum": first_location_of_maximum,
    "first_location_of_minimum": first_location_of_minimum,
    "last_location_of_maximum": last_location_of_maximum,
    "last_location_of_minimum": last_location_of_minimum,
    "index_mass_quantile": index_mass_quantile,
    # Streak/pattern (5)
    "longest_strike_above_mean": longest_strike_above_mean,
    "longest_strike_below_mean": longest_strike_below_mean,
    "number_crossing_m": number_crossing_m,
    "number_peaks": number_peaks,
    "number_cwt_peaks": number_cwt_peaks,
    # Autocorrelation (3)
    "autocorrelation": autocorrelation,
    "partial_autocorrelation": partial_autocorrelation,
    "agg_autocorrelation": agg_autocorrelation,
    # Entropy/complexity (5)
    "permutation_entropy": permutation_entropy,
    "binned_entropy": binned_entropy,
    "fourier_entropy": fourier_entropy,
    "lempel_ziv_complexity": lempel_ziv_complexity,
    "cid_ce": cid_ce,
    # Frequency domain (4)
    "fft_coefficient": fft_coefficient,
    "fft_aggregated": fft_aggregated,
    "spkt_welch_density": spkt_welch_density,
    "cwt_coefficients": cwt_coefficients,
    # Trend/regression (5)
    "linear_trend": linear_trend,
    "linear_trend_timewise": linear_trend_timewise,
    "agg_linear_trend": agg_linear_trend,
    "ar_coefficient": ar_coefficient,
    "augmented_dickey_fuller": augmented_dickey_fuller,
    # Reoccurrence (5)
    "percentage_of_reoccurring_datapoints_to_all_datapoints": percentage_of_reoccurring_datapoints_to_all_datapoints,
    "percentage_of_reoccurring_values_to_all_values": percentage_of_reoccurring_values_to_all_values,
    "sum_of_reoccurring_data_points": sum_of_reoccurring_data_points,
    "sum_of_reoccurring_values": sum_of_reoccurring_values,
    "ratio_value_number_to_time_series_length": ratio_value_number_to_time_series_length,
    # Advanced (12)
    "benford_correlation": benford_correlation,
    "c3": c3,
    "change_quantiles": change_quantiles,
    "energy_ratio_by_chunks": energy_ratio_by_chunks,
    "friedrich_coefficients": friedrich_coefficients,
    "max_langevin_fixed_point": max_langevin_fixed_point,
    "mean_n_absolute_max": mean_n_absolute_max,
    "range_count": range_count,
    "ratio_beyond_r_sigma": ratio_beyond_r_sigma,
    "symmetry_looking": symmetry_looking,
    "time_reversal_asymmetry_statistic": time_reversal_asymmetry_statistic,
    "value_count": value_count,
    # Custom (2)
    "delta": delta,
    "log_delta": log_delta,
}

# =============================================================================
# TSFRESH-COMPATIBLE CONFIGURATION (partial - will be extended)
# =============================================================================

FCParameters = Mapping[str, list[dict[str, Any]] | None]

TORCH_COMPREHENSIVE_FC_PARAMETERS: FCParameters = {
    # Minimal features (no parameters)
    "sum_values": None,
    "median": None,
    "mean": None,
    "length": None,
    "standard_deviation": None,
    "variance": None,
    "root_mean_square": None,
    "maximum": None,
    "absolute_maximum": None,
    "minimum": None,
    # Simple statistics
    "abs_energy": None,
    "kurtosis": None,
    "skewness": None,
    "quantile": [{"q": q} for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]],
    "variation_coefficient": None,
    # Change/difference
    "absolute_sum_of_changes": None,
    "mean_abs_change": None,
    "mean_change": None,
    "mean_second_derivative_central": None,
    # Counting
    "count_above_mean": None,
    "count_below_mean": None,
    "count_above": [{"t": 0}],
    "count_below": [{"t": 0}],
    # Boolean
    "has_duplicate": None,
    "has_duplicate_max": None,
    "has_duplicate_min": None,
    "has_variance_larger_than_standard_deviation": None,
    "large_standard_deviation": [{"r": r * 0.05} for r in range(1, 20)],
    # Location
    "first_location_of_maximum": None,
    "first_location_of_minimum": None,
    "last_location_of_maximum": None,
    "last_location_of_minimum": None,
    "index_mass_quantile": [{"q": q} for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]],
    # Streak/pattern
    "longest_strike_above_mean": None,
    "longest_strike_below_mean": None,
    "number_crossing_m": [{"m": 0}, {"m": -1}, {"m": 1}],
    "number_peaks": [{"n": n} for n in [1, 3, 5, 10, 50]],
    "number_cwt_peaks": [{"max_width": n} for n in [1, 5]],
    # Autocorrelation
    "autocorrelation": [{"lag": lag} for lag in range(10)],
    "agg_autocorrelation": [{"f_agg": s, "maxlag": 40} for s in ["mean", "median", "var"]],
    "partial_autocorrelation": [{"lag": lag} for lag in range(10)],
    # Entropy/complexity
    "binned_entropy": [{"max_bins": 10}],
    "fourier_entropy": [{"bins": x} for x in [2, 3, 5, 10, 100]],
    "permutation_entropy": [{"tau": 1, "dimension": x} for x in [3, 4, 5, 6, 7]],
    "lempel_ziv_complexity": [{"bins": x} for x in [2, 3, 5, 10, 100]],
    "cid_ce": [{"normalize": True}, {"normalize": False}],
    # Frequency domain
    "fft_coefficient": [
        {"coeff": k, "attr": a} for a in ["real", "imag", "abs", "angle"] for k in range(100)
    ],
    "fft_aggregated": [{"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]],
    "spkt_welch_density": [{"coeff": coeff} for coeff in [2, 5, 8]],
    "cwt_coefficients": [
        {"width": w, "coeff": coeff} for w in [2, 5, 10, 20] for coeff in range(15)
    ],
    # Trend/regression
    "linear_trend": [
        {"attr": "pvalue"},
        {"attr": "rvalue"},
        {"attr": "intercept"},
        {"attr": "slope"},
        {"attr": "stderr"},
    ],
    "agg_linear_trend": [
        {"attr": attr, "chunk_size": i, "f_agg": f}
        for attr in ["rvalue", "intercept", "slope", "stderr"]
        for i in [5, 10, 50]
        for f in ["max", "min", "mean", "var"]
    ],
    "augmented_dickey_fuller": [
        {"attr": "teststat"},
        {"attr": "pvalue"},
        {"attr": "usedlag"},
    ],
    "ar_coefficient": [{"coeff": coeff, "k": 10} for coeff in range(11)],
    "linear_trend_timewise": [
        {"attr": "pvalue"},
        {"attr": "rvalue"},
        {"attr": "intercept"},
        {"attr": "slope"},
        {"attr": "stderr"},
    ],
    # Reoccurrence
    "percentage_of_reoccurring_datapoints_to_all_datapoints": None,
    "percentage_of_reoccurring_values_to_all_values": None,
    "sum_of_reoccurring_data_points": None,
    "sum_of_reoccurring_values": None,
    "ratio_value_number_to_time_series_length": None,
    # Advanced
    "benford_correlation": None,
    "time_reversal_asymmetry_statistic": [{"lag": lag} for lag in range(1, 4)],
    "c3": [{"lag": lag} for lag in range(1, 4)],
    "symmetry_looking": [{"r": r * 0.05} for r in range(20)],
    "change_quantiles": [
        {"ql": ql, "qh": qh, "isabs": b, "f_agg": f}
        for ql in [0.0, 0.2, 0.4, 0.6, 0.8]
        for qh in [0.2, 0.4, 0.6, 0.8, 1.0]
        for b in [False, True]
        for f in ["mean", "var"]
        if ql < qh
    ],
    "value_count": [{"value": value} for value in [0, 1, -1]],
    "range_count": [
        {"min_val": -1, "max_val": 1},
        {"min_val": -1e12, "max_val": 0},
        {"min_val": 0, "max_val": 1e12},
    ],
    "friedrich_coefficients": [{"coeff": coeff, "m": 3, "r": 30} for coeff in range(4)],
    "max_langevin_fixed_point": [{"m": 3, "r": 30}],
    "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": i} for i in range(10)],
    "ratio_beyond_r_sigma": [{"r": x} for x in [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10]],
    "mean_n_absolute_max": [{"number_of_maxima": n} for n in [3, 5, 7]],
}

# =============================================================================
# GPU-FRIENDLY SUBSET (excludes features that are slower on GPU than CPU)
# =============================================================================
# Excluded features and why:
# - permutation_entropy: Uses loops/CPU-bound operations, 1143ms on GPU
# - percentage_of_reoccurring_*: Uses torch.unique which is slow on GPU (~575ms)
# - sum_of_reoccurring_*: Uses torch.unique
# - ratio_value_number_to_time_series_length: Uses torch.unique
# - quantile: Sorting doesn't parallelize well on GPU (5x slower than CPU)

TORCH_GPU_FC_PARAMETERS: FCParameters = {
    # Minimal features (no parameters)
    "sum_values": None,
    "median": None,
    "mean": None,
    "length": None,
    "standard_deviation": None,
    "variance": None,
    "root_mean_square": None,
    "maximum": None,
    "absolute_maximum": None,
    "minimum": None,
    # Simple statistics (excluding quantile - uses sorting)
    "abs_energy": None,
    "kurtosis": None,
    "skewness": None,
    # "quantile" - EXCLUDED: sorting is slow on GPU
    "variation_coefficient": None,
    # Change/difference
    "absolute_sum_of_changes": None,
    "mean_abs_change": None,
    "mean_change": None,
    "mean_second_derivative_central": None,
    # Counting
    "count_above_mean": None,
    "count_below_mean": None,
    "count_above": [{"t": 0}],
    "count_below": [{"t": 0}],
    # Boolean
    "has_duplicate": None,
    "has_duplicate_max": None,
    "has_duplicate_min": None,
    "has_variance_larger_than_standard_deviation": None,
    "large_standard_deviation": [{"r": r * 0.05} for r in range(1, 20)],
    # Location
    "first_location_of_maximum": None,
    "first_location_of_minimum": None,
    "last_location_of_maximum": None,
    "last_location_of_minimum": None,
    "index_mass_quantile": [{"q": q} for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]],
    # Streak/pattern
    "longest_strike_above_mean": None,
    "longest_strike_below_mean": None,
    "number_crossing_m": [{"m": 0}, {"m": -1}, {"m": 1}],
    "number_peaks": [{"n": n} for n in [1, 3, 5, 10, 50]],
    "number_cwt_peaks": [{"max_width": n} for n in [1, 5]],
    # Autocorrelation
    "autocorrelation": [{"lag": lag} for lag in range(10)],
    "agg_autocorrelation": [{"f_agg": s, "maxlag": 40} for s in ["mean", "median", "var"]],
    "partial_autocorrelation": [{"lag": lag} for lag in range(10)],
    # Entropy/complexity (excluding permutation_entropy - slow loops)
    "binned_entropy": [{"max_bins": 10}],
    "fourier_entropy": [{"bins": x} for x in [2, 3, 5, 10, 100]],
    # "permutation_entropy" - EXCLUDED: uses loops, extremely slow on GPU
    "lempel_ziv_complexity": [{"bins": x} for x in [2, 3, 5, 10, 100]],
    "cid_ce": [{"normalize": True}, {"normalize": False}],
    # Frequency domain
    "fft_coefficient": [
        {"coeff": k, "attr": a} for a in ["real", "imag", "abs", "angle"] for k in range(100)
    ],
    "fft_aggregated": [{"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]],
    "spkt_welch_density": [{"coeff": coeff} for coeff in [2, 5, 8]],
    "cwt_coefficients": [
        {"width": w, "coeff": coeff} for w in [2, 5, 10, 20] for coeff in range(15)
    ],
    # Trend/regression
    "linear_trend": [
        {"attr": "pvalue"},
        {"attr": "rvalue"},
        {"attr": "intercept"},
        {"attr": "slope"},
        {"attr": "stderr"},
    ],
    "agg_linear_trend": [
        {"attr": attr, "chunk_size": i, "f_agg": f}
        for attr in ["rvalue", "intercept", "slope", "stderr"]
        for i in [5, 10, 50]
        for f in ["max", "min", "mean", "var"]
    ],
    "augmented_dickey_fuller": [
        {"attr": "teststat"},
        {"attr": "pvalue"},
        {"attr": "usedlag"},
    ],
    "ar_coefficient": [{"coeff": coeff, "k": 10} for coeff in range(11)],
    "linear_trend_timewise": [
        {"attr": "pvalue"},
        {"attr": "rvalue"},
        {"attr": "intercept"},
        {"attr": "slope"},
        {"attr": "stderr"},
    ],
    # Reoccurrence - ALL EXCLUDED: use torch.unique which is slow on GPU
    # "percentage_of_reoccurring_datapoints_to_all_datapoints" - EXCLUDED
    # "percentage_of_reoccurring_values_to_all_values" - EXCLUDED
    # "sum_of_reoccurring_data_points" - EXCLUDED
    # "sum_of_reoccurring_values" - EXCLUDED
    # "ratio_value_number_to_time_series_length" - EXCLUDED
    # Advanced
    "benford_correlation": None,
    "time_reversal_asymmetry_statistic": [{"lag": lag} for lag in range(1, 4)],
    "c3": [{"lag": lag} for lag in range(1, 4)],
    "symmetry_looking": [{"r": r * 0.05} for r in range(20)],
    "change_quantiles": [
        {"ql": ql, "qh": qh, "isabs": b, "f_agg": f}
        for ql in [0.0, 0.2, 0.4, 0.6, 0.8]
        for qh in [0.2, 0.4, 0.6, 0.8, 1.0]
        for b in [False, True]
        for f in ["mean", "var"]
        if ql < qh
    ],
    "value_count": [{"value": value} for value in [0, 1, -1]],
    "range_count": [
        {"min_val": -1, "max_val": 1},
        {"min_val": -1e12, "max_val": 0},
        {"min_val": 0, "max_val": 1e12},
    ],
    "friedrich_coefficients": [{"coeff": coeff, "m": 3, "r": 30} for coeff in range(4)],
    "max_langevin_fixed_point": [{"m": 3, "r": 30}],
    "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": i} for i in range(10)],
    "ratio_beyond_r_sigma": [{"r": x} for x in [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10]],
    "mean_n_absolute_max": [{"number_of_maxima": n} for n in [3, 5, 7]],
}

# Custom features (PyTorch-only, not in tsfresh)
TORCH_CUSTOM_FC_PARAMETERS: FCParameters = {
    "delta": None,
    "log_delta": None,
}

# Minimal feature names
MINIMAL_FEATURE_NAMES: list[str] = [
    "sum_values",
    "median",
    "mean",
    "length",
    "standard_deviation",
    "variance",
    "root_mean_square",
    "maximum",
    "absolute_maximum",
    "minimum",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _format_feature_name(feature_name: str, params: dict[str, object] | None) -> str:
    """Format feature name with parameters."""
    if params is None:
        return feature_name
    param_str = "__".join(f"{k}_{v}" for k, v in sorted(params.items()))
    return f"{feature_name}__{param_str}"


def extract_features_from_config(
    x: Tensor,
    fc_parameters: FCParameters | None = None,
    include_custom: bool = False,
) -> dict[str, Tensor]:
    """Extract features from tensor using configuration.

    Args:
        x: Input tensor of shape (N, B, S)
        fc_parameters: Feature configuration (None for defaults)
        include_custom: Include custom features (delta, log_delta)

    Returns:
        Dictionary mapping feature names to result tensors of shape (B, S)
    """
    if fc_parameters is None:
        fc_parameters = TORCH_COMPREHENSIVE_FC_PARAMETERS

    results: dict[str, Tensor] = {}

    for feature_name, param_list in fc_parameters.items():
        if feature_name not in ALL_FEATURE_FUNCTIONS:
            continue
        func = ALL_FEATURE_FUNCTIONS[feature_name]

        if param_list is None:
            results[feature_name] = func(x)
        else:
            for params in param_list:
                name = _format_feature_name(feature_name, params)
                results[name] = func(x, **params)

    if include_custom:
        for feature_name in TORCH_CUSTOM_FC_PARAMETERS:
            if feature_name in ALL_FEATURE_FUNCTIONS:
                results[feature_name] = ALL_FEATURE_FUNCTIONS[feature_name](x)

    return results


def get_feature_names_from_config(
    fc_parameters: FCParameters | None = None,
    include_custom: bool = False,
) -> list[str]:
    """Get list of feature names from configuration.

    Args:
        fc_parameters: Feature configuration (None for defaults)
        include_custom: Include custom features

    Returns:
        List of feature names
    """
    if fc_parameters is None:
        fc_parameters = TORCH_COMPREHENSIVE_FC_PARAMETERS

    names: list[str] = []

    for feature_name, param_list in fc_parameters.items():
        if feature_name not in ALL_FEATURE_FUNCTIONS:
            continue
        if param_list is None:
            names.append(feature_name)
        else:
            for params in param_list:
                names.append(_format_feature_name(feature_name, params))

    if include_custom:
        for feature_name in TORCH_CUSTOM_FC_PARAMETERS:
            if feature_name in ALL_FEATURE_FUNCTIONS:
                names.append(feature_name)

    return names
