# pyright: basic
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


@torch.no_grad()
def mean_n_absolute_max(x: Tensor, number_of_maxima: int = 1) -> Tensor:
    """Mean of n largest absolute values (optimized with topk)."""
    x_abs = x.abs()
    k = min(number_of_maxima, x.shape[0])
    # Use topk instead of full sort - much faster for small k
    top_vals, _ = x_abs.topk(k, dim=0)
    return top_vals.mean(dim=0)


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


# =============================================================================
# BATCHED STATISTICAL FEATURES
# =============================================================================


@torch.no_grad()
def quantile_batched(x: Tensor, qs: list[float]) -> Tensor:
    """Compute multiple quantiles at once.

    Args:
        x: Input tensor of shape (N, B, S)
        qs: List of quantile values (0.0 to 1.0)

    Returns:
        Tensor of shape (len(qs), B, S)
    """
    q_tensor = torch.tensor(qs, dtype=x.dtype, device=x.device)
    return torch.quantile(x, q_tensor, dim=0)


@torch.no_grad()
def large_standard_deviation_batched(x: Tensor, rs: list[float]) -> Tensor:
    """Check if std > r * range for multiple r values at once.

    Args:
        x: Input tensor of shape (N, B, S)
        rs: List of r threshold values

    Returns:
        Tensor of shape (len(rs), B, S)
    """
    std = x.std(dim=0, correction=0)
    range_val = x.max(dim=0).values - x.min(dim=0).values

    results = []
    for r in rs:
        results.append((std > r * range_val).float())

    return torch.stack(results, dim=0)


@torch.no_grad()
def symmetry_looking_batched(x: Tensor, rs: list[float]) -> Tensor:
    """Check if distribution looks symmetric for multiple r values.

    Args:
        x: Input tensor of shape (N, B, S)
        rs: List of r threshold values

    Returns:
        Tensor of shape (len(rs), B, S)
    """
    mu = x.mean(dim=0)
    med = x.median(dim=0).values
    range_val = x.max(dim=0).values - x.min(dim=0).values

    results = []
    for r in rs:
        results.append(((mu - med).abs() < r * range_val).float())

    return torch.stack(results, dim=0)


@torch.no_grad()
def ratio_beyond_r_sigma_batched(x: Tensor, rs: list[float]) -> Tensor:
    """Compute ratio of values beyond r standard deviations for multiple r.

    Args:
        x: Input tensor of shape (N, B, S)
        rs: List of r multiplier values

    Returns:
        Tensor of shape (len(rs), B, S)
    """
    mu = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, correction=0, keepdim=True)
    deviation = (x - mu).abs()

    results = []
    for r in rs:
        beyond = deviation > r * std
        results.append(beyond.float().mean(dim=0))

    return torch.stack(results, dim=0)


@torch.no_grad()
def mean_n_absolute_max_batched(x: Tensor, ns: list[int]) -> Tensor:
    """Compute mean_n_absolute_max for multiple n values at once.

    Args:
        x: Input tensor of shape (N, B, S)
        ns: List of number_of_maxima values

    Returns:
        Tensor of shape (len(ns), B, S)
    """
    batch_size, n_states = x.shape[1], x.shape[2]
    x_abs = x.abs()
    max_n = x.shape[0]

    if not ns:
        return torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device)

    max_k = min(max(ns), max_n)
    top_vals, _ = x_abs.topk(max_k, dim=0)

    results = []
    for n in ns:
        k = min(n, max_n)
        results.append(top_vals[:k].mean(dim=0))

    return torch.stack(results, dim=0)
