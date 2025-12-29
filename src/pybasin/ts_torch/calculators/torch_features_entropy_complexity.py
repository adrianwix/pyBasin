# pyright: basic
import math

import torch
from torch import Tensor

# =============================================================================
# ENTROPY/COMPLEXITY FEATURES (5 features)
# =============================================================================


@torch.no_grad()
def permutation_entropy(x: Tensor, tau: int = 1, dimension: int = 3) -> Tensor:
    """Permutation entropy (fully vectorized GPU implementation)."""
    n, batch_size, n_states = x.shape
    num_patterns = n - (dimension - 1) * tau

    if num_patterns <= 0:
        return torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)

    indices = (
        torch.arange(num_patterns, device=x.device).unsqueeze(1)
        + torch.arange(dimension, device=x.device).unsqueeze(0) * tau
    )
    embedded = x[indices]  # (num_patterns, dimension, B, S)

    ranks = embedded.argsort(dim=1)

    multipliers = torch.tensor(
        [math.factorial(dimension - 1 - i) for i in range(dimension)],
        device=x.device,
        dtype=torch.long,
    ).view(1, dimension, 1, 1)
    pattern_ids = (ranks * multipliers).sum(dim=1)  # (num_patterns, B, S)

    n_permutations = math.factorial(dimension)
    pattern_ids_flat = pattern_ids.permute(1, 2, 0).reshape(-1, num_patterns)  # (B*S, num_patterns)

    one_hot = torch.zeros(batch_size * n_states, n_permutations, dtype=x.dtype, device=x.device)
    one_hot.scatter_add_(
        dim=1,
        index=pattern_ids_flat,
        src=torch.ones_like(pattern_ids_flat, dtype=x.dtype),
    )
    counts = one_hot  # (B*S, n_permutations)

    probs = counts / num_patterns
    log_probs = torch.where(probs > 0, probs.log(), torch.zeros_like(probs))
    entropy = -(probs * log_probs).sum(dim=1)

    max_entropy = math.log(n_permutations)
    result = entropy / max_entropy if max_entropy > 0 else entropy

    return result.reshape(batch_size, n_states)


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


# =============================================================================
# BATCHED ENTROPY FEATURES
# =============================================================================


@torch.no_grad()
def fourier_entropy_batched(x: Tensor, bins_list: list[int]) -> Tensor:
    """Compute fourier_entropy for multiple bins values at once.

    Computes FFT and PSD once, then returns entropy for each bins value.
    Note: The bins parameter is not actually used in the tsfresh implementation
    (it's always spectral entropy), so this returns the same value for all bins.

    Args:
        x: Input tensor of shape (N, B, S)
        bins_list: List of bins values (not actually used in computation)

    Returns:
        Tensor of shape (len(bins_list), B, S)
    """
    fft_result = torch.fft.rfft(x, dim=0)
    psd = fft_result.real**2 + fft_result.imag**2
    psd_sum = psd.sum(dim=0, keepdim=True)
    psd_norm = psd / (psd_sum + 1e-10)
    psd_norm = psd_norm.clamp(min=1e-10)
    entropy = -(psd_norm * psd_norm.log()).sum(dim=0)

    return entropy.unsqueeze(0).expand(len(bins_list), -1, -1).clone()


@torch.no_grad()
def lempel_ziv_complexity(x: Tensor, bins: int = 2) -> Tensor:
    """Lempel-Ziv complexity approximation (optimized)."""
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


@torch.no_grad()
def approximate_entropy(x: Tensor, m: int = 2, r: float = 0.3) -> Tensor:
    """Approximate entropy of the time series."""
    raise NotImplementedError(
        "approximate_entropy is not yet implemented in PyTorch. "
        "This feature is excluded from tsfresh's EfficientFCParameters due to high computational cost."
    )


@torch.no_grad()
def sample_entropy(x: Tensor, m: int = 2, r: float = 0.3) -> Tensor:
    """Sample entropy of the time series."""
    raise NotImplementedError(
        "sample_entropy is not yet implemented in PyTorch. "
        "This feature is excluded from tsfresh's EfficientFCParameters due to high computational cost."
    )
