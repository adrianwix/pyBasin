# pyright: basic
import torch
from torch import Tensor

from pybasin.ts_torch.torch_feature_utilities import local_maxima_1d

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
    """Partial autocorrelation at given lag using Durbin-Levinson (fully vectorized)."""
    if lag == 0:
        return torch.ones(x.shape[1:], dtype=x.dtype, device=x.device)

    n = x.shape[0]
    batch_size = x.shape[1]
    n_states = x.shape[2]
    maxlag = min(lag, n - 1)

    if maxlag < 1:
        return torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)

    x_centered = x - x.mean(dim=0, keepdim=True)
    n_fft = 2 * n
    fft_x = torch.fft.rfft(x_centered, n=n_fft, dim=0)
    power_spectrum = fft_x.real**2 + fft_x.imag**2
    autocorr_full = torch.fft.irfft(power_spectrum, n=n_fft, dim=0)

    var = autocorr_full[0:1]
    acf = autocorr_full[: maxlag + 1] / (var + 1e-10)  # (maxlag+1, B, S)

    phi = torch.zeros(maxlag + 1, batch_size, n_states, dtype=x.dtype, device=x.device)
    phi[1] = acf[1]

    for k in range(2, maxlag + 1):
        phi_prev = phi[1:k].flip(dims=[0])
        acf_prev = acf[1:k]
        num = acf[k] - (phi[1:k] * acf_prev.flip(dims=[0])).sum(dim=0)
        denom = 1.0 - (phi[1:k] * acf_prev).sum(dim=0)
        phi_k = num / (denom + 1e-10)
        phi[k] = phi_k
        phi[1:k] = phi[1:k] - phi_k.unsqueeze(0) * phi_prev

    if lag <= maxlag:
        return phi[lag]
    else:
        return torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)


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
# BATCHED AUTOCORRELATION FEATURES
# =============================================================================


@torch.no_grad()
def autocorrelation_batched(x: Tensor, lags: list[int]) -> Tensor:
    """Compute autocorrelation for multiple lags at once using FFT.

    Args:
        x: Input tensor of shape (N, B, S)
        lags: List of lag values to compute

    Returns:
        Tensor of shape (len(lags), B, S) with autocorrelation at each lag
    """
    n = x.shape[0]
    batch_size, n_states = x.shape[1], x.shape[2]
    max_lag = max(lags) if lags else 0

    if max_lag >= n:
        max_lag = n - 1
        lags = [lag for lag in lags if lag < n]

    if not lags:
        return torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device)

    x_centered = x - x.mean(dim=0, keepdim=True)
    var = (x_centered**2).sum(dim=0)

    n_fft = 2 * n
    fft_x = torch.fft.rfft(x_centered, n=n_fft, dim=0)
    power_spectrum = fft_x.real**2 + fft_x.imag**2
    autocorr_full = torch.fft.irfft(power_spectrum, n=n_fft, dim=0)

    autocorr_normalized = autocorr_full / (var.unsqueeze(0) + 1e-10)

    results = []
    for lag in lags:
        if lag == 0:
            results.append(torch.ones(batch_size, n_states, dtype=x.dtype, device=x.device))
        elif lag < n:
            results.append(autocorr_normalized[lag])
        else:
            results.append(torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device))

    return torch.stack(results, dim=0)


@torch.no_grad()
def partial_autocorrelation_batched(x: Tensor, lags: list[int]) -> Tensor:
    """Compute partial autocorrelation for multiple lags at once.

    Computes Durbin-Levinson algorithm once for max(lags), then extracts
    requested lag values. Stores PACF values separately from AR coefficients
    since the algorithm modifies AR coefficients in-place.

    Args:
        x: Input tensor of shape (N, B, S)
        lags: List of lag values to compute

    Returns:
        Tensor of shape (len(lags), B, S) with PACF at each lag
    """
    n, batch_size, n_states = x.shape

    if not lags:
        return torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device)

    maxlag = min(max(lags), n - 1)

    if maxlag < 1:
        results = []
        for lag in lags:
            if lag == 0:
                results.append(torch.ones(batch_size, n_states, dtype=x.dtype, device=x.device))
            else:
                results.append(torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device))
        return torch.stack(results, dim=0)

    x_centered = x - x.mean(dim=0, keepdim=True)
    n_fft = 2 * n
    fft_x = torch.fft.rfft(x_centered, n=n_fft, dim=0)
    power_spectrum = fft_x.real**2 + fft_x.imag**2
    autocorr_full = torch.fft.irfft(power_spectrum, n=n_fft, dim=0)

    var = autocorr_full[0:1]
    acf = autocorr_full[: maxlag + 1] / (var + 1e-10)

    phi = torch.zeros(maxlag + 1, batch_size, n_states, dtype=x.dtype, device=x.device)
    pacf = torch.zeros(maxlag + 1, batch_size, n_states, dtype=x.dtype, device=x.device)
    pacf[0] = 1.0
    phi[1] = acf[1]
    pacf[1] = acf[1]

    for k in range(2, maxlag + 1):
        phi_prev = phi[1:k].flip(dims=[0])
        acf_prev = acf[1:k]
        num = acf[k] - (phi[1:k] * acf_prev.flip(dims=[0])).sum(dim=0)
        denom = 1.0 - (phi[1:k] * acf_prev).sum(dim=0)
        phi_k = num / (denom + 1e-10)
        pacf[k] = phi_k
        phi[k] = phi_k
        phi[1:k] = phi[1:k] - phi_k.unsqueeze(0) * phi_prev

    results = []
    for lag in lags:
        if lag == 0:
            results.append(torch.ones(batch_size, n_states, dtype=x.dtype, device=x.device))
        elif 0 < lag <= maxlag:
            results.append(pacf[lag])
        else:
            results.append(torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device))

    return torch.stack(results, dim=0)


@torch.no_grad()
def agg_autocorrelation_batched(x: Tensor, params: list[dict]) -> Tensor:
    """Compute agg_autocorrelation for multiple (maxlag, f_agg) combinations at once.

    Groups by maxlag to minimize FFT computations.

    Args:
        x: Input tensor of shape (N, B, S)
        params: List of parameter dicts, each with keys:
            - "maxlag": int (defaults to 40 if not specified)
            - "f_agg": str ("mean", "median", "var")

    Returns:
        Tensor of shape (len(params), B, S)
    """
    n, batch_size, n_states = x.shape

    if not params:
        return torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device)

    groups: dict[int, list[tuple[int, str]]] = {}
    for idx, p in enumerate(params):
        maxlag = p.get("maxlag", 40)
        f_agg = p.get("f_agg", "mean")
        groups.setdefault(maxlag, []).append((idx, f_agg))

    results = torch.zeros(len(params), batch_size, n_states, dtype=x.dtype, device=x.device)

    x_centered = x - x.mean(dim=0, keepdim=True)
    n_fft = 2 * n
    fft_x = torch.fft.rfft(x_centered, n=n_fft, dim=0)
    power_spectrum = fft_x.real**2 + fft_x.imag**2
    autocorr_full = torch.fft.irfft(power_spectrum, n=n_fft, dim=0)
    var = autocorr_full[0:1]

    for maxlag, items in groups.items():
        maxlag = min(maxlag, n - 1)
        acf = autocorr_full[1 : maxlag + 1] / (var + 1e-10)

        for param_idx, f_agg in items:
            if f_agg == "mean":
                results[param_idx] = acf.mean(dim=0)
            elif f_agg == "median":
                results[param_idx] = acf.median(dim=0).values
            elif f_agg == "var":
                results[param_idx] = acf.var(dim=0, correction=0)
            else:
                results[param_idx] = acf.mean(dim=0)

    return results


@torch.no_grad()
def autocorrelation_periodicity(
    x: Tensor, min_lag: int = 2, peak_threshold: float = 0.3, output: str = "strength"
) -> Tensor:
    """Compute autocorrelation-based periodicity measures.

    TODO: Support returning multiple outputs (K, B, S) instead of requiring separate calls.
          This would require updating torch_feature_extractor.py and torch_feature_processors.py
          to handle 3D output tensors properly.

    Returns either the periodicity strength (height of first significant autocorrelation
    peak) or the period estimate (lag of that peak). This is useful for detecting
    limit cycles vs chaos vs fixed points.

    Uses FFT for efficient autocorrelation computation and local_maxima_1d for
    robust peak detection.

    Args:
        x: Input tensor of shape (N, B, S) where N is timesteps, B is batch, S is states.
        min_lag: Minimum lag to search for peaks (to skip lag-0 peak). Default 2.
        peak_threshold: Minimum autocorrelation value to consider as a peak. Default 0.3.
        output: Which value to return - "strength" or "period". Default "strength".

    Returns:
        Tensor of shape (B, S) with either periodicity strength or period estimate.
    """

    n, batch_size, n_states = x.shape

    x_centered = x - x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True)
    x_normalized = x_centered / (x_std + 1e-10)

    n_fft = 2 * n
    fft_x = torch.fft.rfft(x_normalized, n=n_fft, dim=0)
    power_spectrum = fft_x.real**2 + fft_x.imag**2
    autocorr_full = torch.fft.irfft(power_spectrum, n=n_fft, dim=0)

    autocorr = autocorr_full[:n] / (autocorr_full[0:1] + 1e-10)

    peaks_mask = local_maxima_1d(autocorr)

    peaks_mask[:min_lag] = False

    above_threshold = autocorr > peak_threshold
    valid_peaks = peaks_mask & above_threshold

    periodicity_strength = torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device)
    period_estimate = torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device)

    for b in range(batch_size):
        for s in range(n_states):
            peak_indices = torch.where(valid_peaks[:, b, s])[0]
            if len(peak_indices) > 0:
                first_peak_idx = peak_indices[0]
                periodicity_strength[b, s] = autocorr[first_peak_idx, b, s]
                period_estimate[b, s] = first_peak_idx.float()

    if output == "period":
        return period_estimate
    return periodicity_strength
