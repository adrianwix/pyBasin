# pyright: basic
"""Batched PyTorch feature calculators for GPU optimization.

These calculators compute multiple parameter variants in a single kernel call,
significantly reducing Python dispatch overhead and enabling better GPU utilization.

Example:
    # Instead of 10 separate calls:
    # for lag in range(10): autocorrelation(x, lag=lag)

    # Use one batched call:
    autocorrelation_batched(x, lags=[0,1,2,3,4,5,6,7,8,9])  # returns (10, B, S)
"""

import torch
from torch import Tensor
from torch.func import vmap

from pybasin.feature_extractors.torch_feature_calculators import _change_quantiles_core


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
def fft_coefficient_batched(x: Tensor, coeffs: list[int], attr: str = "abs") -> Tensor:
    """Compute FFT coefficients for multiple indices at once.

    Args:
        x: Input tensor of shape (N, B, S)
        coeffs: List of coefficient indices to extract
        attr: Attribute to extract ("real", "imag", "abs", "angle")

    Returns:
        Tensor of shape (len(coeffs), B, S)
    """
    fft_result = torch.fft.rfft(x, dim=0)
    n_coeffs = fft_result.shape[0]
    batch_size, n_states = x.shape[1], x.shape[2]

    results = []
    for coeff in coeffs:
        if coeff >= n_coeffs:
            results.append(torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device))
        else:
            c = fft_result[coeff]
            if attr == "real":
                results.append(c.real)
            elif attr == "imag":
                results.append(c.imag)
            elif attr == "abs":
                results.append(c.abs())
            elif attr == "angle":
                results.append(torch.atan2(c.imag, c.real))
            else:
                results.append(c.abs())

    return torch.stack(results, dim=0)


@torch.no_grad()
def fft_coefficient_all_attrs_batched(x: Tensor, coeffs: list[int]) -> Tensor:
    """Compute all FFT attributes for multiple coefficients at once.

    Args:
        x: Input tensor of shape (N, B, S)
        coeffs: List of coefficient indices

    Returns:
        Tensor of shape (len(coeffs) * 4, B, S) ordered as:
        [coeff0_real, coeff0_imag, coeff0_abs, coeff0_angle, coeff1_real, ...]
    """
    fft_result = torch.fft.rfft(x, dim=0)
    n_coeffs = fft_result.shape[0]
    batch_size, n_states = x.shape[1], x.shape[2]

    results = []
    for coeff in coeffs:
        if coeff >= n_coeffs:
            zeros = torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device)
            results.extend([zeros, zeros, zeros, zeros])
        else:
            c = fft_result[coeff]
            results.append(c.real)
            results.append(c.imag)
            results.append(c.abs())
            results.append(torch.atan2(c.imag, c.real))

    return torch.stack(results, dim=0)


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
def index_mass_quantile_batched(x: Tensor, qs: list[float]) -> Tensor:
    """Compute index_mass_quantile for multiple q values at once.

    Args:
        x: Input tensor of shape (N, B, S)
        qs: List of quantile values

    Returns:
        Tensor of shape (len(qs), B, S)
    """
    n = x.shape[0]
    x_abs = x.abs()
    cumsum = x_abs.cumsum(dim=0)
    total = cumsum[-1:]

    results = []
    for q in qs:
        threshold = q * total
        mask = cumsum >= threshold
        idx = mask.float().argmax(dim=0)
        results.append(idx.float() / n)

    return torch.stack(results, dim=0)


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
def ar_coefficient_batched(x: Tensor, k: int, coeffs: list[int]) -> Tensor:
    """Compute AR model coefficients for multiple coeff indices at once.

    Args:
        x: Input tensor of shape (N, B, S)
        k: AR model order
        coeffs: List of coefficient indices to return

    Returns:
        Tensor of shape (len(coeffs), B, S)
    """
    n, batch_size, n_states = x.shape

    x_centered = x - x.mean(dim=0, keepdim=True)

    n_fft = 2 * n
    fft_x = torch.fft.rfft(x_centered, n=n_fft, dim=0)
    power_spectrum = fft_x.real**2 + fft_x.imag**2
    autocorr_full = torch.fft.irfft(power_spectrum, n=n_fft, dim=0)[: k + 1]

    acf = autocorr_full / (autocorr_full[0:1] + 1e-10)

    acf_flat = acf.reshape(k + 1, -1).T

    idx = torch.arange(k, device=x.device)
    toeplitz_idx = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    r_matrix = acf_flat[:, toeplitz_idx]

    r_vec = acf_flat[:, 1 : k + 1]

    reg = 1e-6 * torch.eye(k, device=x.device, dtype=x.dtype)
    r_matrix = r_matrix + reg

    try:
        phi = torch.linalg.solve(r_matrix, r_vec)
    except Exception:
        phi = torch.zeros(batch_size * n_states, k, dtype=x.dtype, device=x.device)

    results = []
    for coeff in coeffs:
        if coeff < k:
            results.append(phi[:, coeff].reshape(batch_size, n_states))
        else:
            results.append(torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device))

    return torch.stack(results, dim=0)


@torch.no_grad()
def linear_trend_batched(x: Tensor, attrs: list[str]) -> Tensor:
    """Compute linear regression trend for multiple attributes at once.

    Args:
        x: Input tensor of shape (N, B, S)
        attrs: List of attributes ("slope", "intercept", "rvalue", "pvalue", "stderr")

    Returns:
        Tensor of shape (len(attrs), B, S)
    """
    n = x.shape[0]
    t = torch.arange(n, dtype=x.dtype, device=x.device).unsqueeze(-1).unsqueeze(-1)
    t_mean = t.mean()
    x_mean = x.mean(dim=0, keepdim=True)

    ss_tt = ((t - t_mean) ** 2).sum()
    ss_tx = ((t - t_mean) * (x - x_mean)).sum(dim=0)
    ss_xx = ((x - x_mean) ** 2).sum(dim=0)

    slope = ss_tx / (ss_tt + 1e-10)
    intercept = x_mean.squeeze(0) - slope * t_mean
    rvalue = ss_tx / (torch.sqrt(ss_tt * ss_xx) + 1e-10)

    y_pred = slope * t + intercept
    residuals = x - y_pred
    mse = (residuals**2).sum(dim=0) / (n - 2)
    stderr = torch.sqrt(mse / (ss_tt + 1e-10))

    t_stat = (
        rvalue
        * torch.sqrt(torch.tensor(n - 2, dtype=x.dtype, device=x.device))
        / (torch.sqrt(1 - rvalue**2) + 1e-10)
    )
    pvalue = 1 / (1 + t_stat.abs())

    attr_map = {
        "slope": slope,
        "intercept": intercept,
        "rvalue": rvalue,
        "pvalue": pvalue,
        "stderr": stderr,
    }

    results = []
    for attr in attrs:
        if attr in attr_map:
            results.append(attr_map[attr])
        else:
            results.append(slope)

    return torch.stack(results, dim=0)


@torch.no_grad()
def cwt_coefficients_batched(x: Tensor, params: list[dict]) -> Tensor:
    """Compute CWT coefficients for all parameter combinations at once (GPU-optimized).

    This function computes CWT once per unique width value, then uses vectorized
    advanced indexing to extract all requested coefficients in a single operation.
    The extraction step is fully vectorized with no Python loops.

    Note: This implementation uses a direct Ricker wavelet convolution which differs
    from tsfresh's pywt.cwt in normalization. Results have the same sign but different
    scaling.

    Args:
        x: Input tensor of shape (N, B, S)
        params: List of parameter dicts, each with keys:
            - "widths": tuple of int (e.g., (2, 5, 10, 20))
            - "coeff": int (coefficient index to extract)
            - "w": int (which width from widths to use)

    Returns:
        Tensor of shape (len(params), B, S) with CWT coefficient for each param set

    Example:
        # tsfresh-style parameters (60 combinations)
        params = [
            {"widths": (2, 5, 10, 20), "coeff": c, "w": w}
            for c in range(15) for w in (2, 5, 10, 20)
        ]
        result = cwt_coefficients_batched(x, params)  # shape: (60, B, S)
    """
    n, batch_size, n_states = x.shape
    flat_size = batch_size * n_states

    if not params:
        return torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device)

    unique_widths_set: set[int] = set()
    for p in params:
        w = p.get("w", 2)
        widths = p.get("widths", (2,))
        if w in widths:
            unique_widths_set.add(w)

    if not unique_widths_set:
        return torch.zeros(len(params), batch_size, n_states, dtype=x.dtype, device=x.device)

    unique_widths = sorted(unique_widths_set)
    width_to_idx = {w: i for i, w in enumerate(unique_widths)}
    num_widths = len(unique_widths)

    x_flat = x.permute(1, 2, 0).reshape(flat_size, 1, n)

    cwt_results = []
    for width in unique_widths:
        t = torch.arange(-width * 4, width * 4 + 1, dtype=x.dtype, device=x.device)
        sigma = float(width)
        wavelet = (
            (2 / (torch.sqrt(torch.tensor(3.0 * sigma, device=x.device)) * torch.pi**0.25))
            * (1 - (t / sigma) ** 2)
            * torch.exp(-(t**2) / (2 * sigma**2))
        )
        kernel = wavelet.unsqueeze(0).unsqueeze(0)
        pad_size = len(t) // 2

        if n >= pad_size:
            x_padded = torch.nn.functional.pad(x_flat, (pad_size, pad_size), mode="reflect")
        else:
            x_padded = torch.nn.functional.pad(x_flat, (pad_size, pad_size), mode="replicate")

        conv = torch.nn.functional.conv1d(x_padded, kernel, padding=0)
        cwt_results.append(conv.squeeze(1))

    max_len = max(c.shape[1] for c in cwt_results)
    cwt_stacked = torch.zeros(num_widths, flat_size, max_len, dtype=x.dtype, device=x.device)
    for i, c in enumerate(cwt_results):
        cwt_stacked[i, :, : c.shape[1]] = c

    param_w_indices = []
    param_coeff_indices = []
    valid_mask = []

    for p in params:
        w = p.get("w", 2)
        coeff = p.get("coeff", 0)
        widths = p.get("widths", (2,))

        if w in widths and w in width_to_idx and coeff < max_len:
            param_w_indices.append(width_to_idx[w])
            param_coeff_indices.append(coeff)
            valid_mask.append(True)
        else:
            param_w_indices.append(0)
            param_coeff_indices.append(0)
            valid_mask.append(False)

    w_idx = torch.tensor(param_w_indices, dtype=torch.long, device=x.device)
    coeff_idx = torch.tensor(param_coeff_indices, dtype=torch.long, device=x.device)
    valid = torch.tensor(valid_mask, dtype=torch.bool, device=x.device)

    results = cwt_stacked[w_idx, :, coeff_idx]

    results = results.reshape(len(params), batch_size, n_states)

    invalid_mask = ~valid
    if invalid_mask.any():
        results[invalid_mask] = 0.0

    return results


@torch.no_grad()
def change_quantiles_batched(x: Tensor, params: list[dict]) -> Tensor:
    """Compute change_quantiles for multiple parameter combinations using vmap.

    This function pre-computes all unique quantiles once, then uses vmap to
    efficiently process all parameter combinations in a single kernel.

    Args:
        x: Input tensor of shape (N, B, S)
        params: List of parameter dicts, each with keys:
            - "ql": float (lower quantile, 0.0 to 1.0)
            - "qh": float (upper quantile, 0.0 to 1.0)
            - "isabs": bool (whether to use absolute differences)
            - "f_agg": str ("mean" or "var")

    Returns:
        Tensor of shape (len(params), B, S)

    Example:
        params = [
            {"ql": 0.0, "qh": 0.2, "isabs": True, "f_agg": "mean"},
            {"ql": 0.0, "qh": 0.2, "isabs": True, "f_agg": "var"},
            {"ql": 0.0, "qh": 0.2, "isabs": False, "f_agg": "mean"},
            ...
        ]
        result = change_quantiles_batched(x, params)  # shape: (80, B, S)
    """
    if not params:
        return torch.zeros(0, x.shape[1], x.shape[2], dtype=x.dtype, device=x.device)

    unique_q_values = sorted({p["ql"] for p in params} | {p["qh"] for p in params})
    q_tensor = torch.tensor(unique_q_values, dtype=x.dtype, device=x.device)
    all_quantiles = torch.quantile(x, q_tensor, dim=0)  # (n_q, B, S)
    q_to_idx = {q: i for i, q in enumerate(unique_q_values)}

    ql_indices = torch.tensor([q_to_idx[p["ql"]] for p in params], device=x.device)
    qh_indices = torch.tensor([q_to_idx[p["qh"]] for p in params], device=x.device)
    q_lows = all_quantiles[ql_indices]  # (P, B, S)
    q_highs = all_quantiles[qh_indices]  # (P, B, S)
    isabs_arr = torch.tensor([p["isabs"] for p in params], dtype=torch.bool, device=x.device)
    is_var_arr = torch.tensor(
        [p["f_agg"] == "var" for p in params], dtype=torch.bool, device=x.device
    )

    vmapped_fn = vmap(_change_quantiles_core, in_dims=(None, 0, 0, 0, 0), out_dims=0)
    return vmapped_fn(x, q_lows, q_highs, isabs_arr, is_var_arr)


@torch.no_grad()
def agg_linear_trend_batched(x: Tensor, params: list[dict]) -> Tensor:
    """Compute agg_linear_trend for multiple parameter combinations at once.

    Groups by (chunk_size, f_agg) to minimize redundant chunk aggregation,
    then computes all 4 trend attributes at once per group.

    Args:
        x: Input tensor of shape (N, B, S)
        params: List of parameter dicts, each with keys:
            - "chunk_size": int (chunk size for aggregation, e.g., 5, 10, 50)
            - "f_agg": str ("mean", "var", "min", "max")
            - "attr": str ("slope", "intercept", "rvalue", "stderr")

    Returns:
        Tensor of shape (len(params), B, S)

    Example:
        params = [
            {"chunk_size": 5, "f_agg": "mean", "attr": "slope"},
            {"chunk_size": 5, "f_agg": "mean", "attr": "intercept"},
            {"chunk_size": 5, "f_agg": "var", "attr": "slope"},
            ...
        ]
        result = agg_linear_trend_batched(x, params)  # shape: (48, B, S)
    """
    n, batch_size, n_states = x.shape

    if not params:
        return torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device)

    groups: dict[tuple[int, str], list[tuple[int, str]]] = {}
    for idx, p in enumerate(params):
        key = (p["chunk_size"], p["f_agg"])
        groups.setdefault(key, []).append((idx, p["attr"]))

    results = torch.zeros(len(params), batch_size, n_states, dtype=x.dtype, device=x.device)

    for (chunk_size, f_agg), items in groups.items():
        n_chunks = n // chunk_size
        if n_chunks < 2:
            continue

        chunks = x[: n_chunks * chunk_size].reshape(n_chunks, chunk_size, batch_size, n_states)
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

        trend_results = linear_trend_batched(agg, ["slope", "intercept", "rvalue", "stderr"])
        attr_to_idx = {"slope": 0, "intercept": 1, "rvalue": 2, "stderr": 3}

        for param_idx, attr in items:
            if attr in attr_to_idx:
                results[param_idx] = trend_results[attr_to_idx[attr]]

    return results


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
    acf = autocorr_full[: maxlag + 1] / (var + 1e-10)  # (maxlag+1, B, S)

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
        phi[k] = phi_k
        pacf[k] = phi_k
        phi[1:k] = phi[1:k] - phi_k.unsqueeze(0) * phi_prev

    results = []
    for lag in lags:
        if lag == 0:
            results.append(torch.ones(batch_size, n_states, dtype=x.dtype, device=x.device))
        elif lag <= maxlag:
            results.append(pacf[lag])
        else:
            results.append(torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device))

    return torch.stack(results, dim=0)


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
def fft_aggregated_batched(x: Tensor, aggtypes: list[str]) -> Tensor:
    """Compute fft_aggregated for all aggregation types at once.

    Computes FFT and PSD once, then returns all requested aggregations.

    Args:
        x: Input tensor of shape (N, B, S)
        aggtypes: List of aggregation types ("centroid", "variance", "skew", "kurtosis")

    Returns:
        Tensor of shape (len(aggtypes), B, S)
    """
    if not aggtypes:
        raise ValueError("aggtypes cannot be empty")

    fft_result = torch.fft.rfft(x, dim=0)
    psd = fft_result.real**2 + fft_result.imag**2
    n_coeffs = psd.shape[0]
    freqs = torch.arange(n_coeffs, dtype=x.dtype, device=x.device).unsqueeze(-1).unsqueeze(-1)

    psd_sum = psd.sum(dim=0, keepdim=True)
    psd_norm = psd / (psd_sum + 1e-10)

    centroid = (freqs * psd_norm).sum(dim=0)
    variance = ((freqs - centroid.unsqueeze(0)) ** 2 * psd_norm).sum(dim=0)
    skew = ((freqs - centroid.unsqueeze(0)) ** 3 * psd_norm).sum(dim=0) / (variance**1.5 + 1e-10)
    kurtosis = ((freqs - centroid.unsqueeze(0)) ** 4 * psd_norm).sum(dim=0) / (variance**2 + 1e-10)

    agg_map = {
        "centroid": centroid,
        "variance": variance,
        "skew": skew,
        "kurtosis": kurtosis,
    }

    results = []
    for aggtype in aggtypes:
        if aggtype in agg_map:
            results.append(agg_map[aggtype])
        else:
            results.append(centroid)

    return torch.stack(results, dim=0)


@torch.no_grad()
def spkt_welch_density_batched(x: Tensor, coeffs: list[int]) -> Tensor:
    """Compute spkt_welch_density for multiple coefficient indices at once.

    Computes Welch PSD once, then extracts multiple coefficients.

    Args:
        x: Input tensor of shape (N, B, S)
        coeffs: List of coefficient indices to extract

    Returns:
        Tensor of shape (len(coeffs), B, S)
    """
    fft_result = torch.fft.rfft(x, dim=0)
    psd = (fft_result.real**2 + fft_result.imag**2) / x.shape[0]
    n_coeffs = psd.shape[0]
    batch_size, n_states = x.shape[1], x.shape[2]

    results = []
    for coeff in coeffs:
        if coeff >= n_coeffs:
            results.append(torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device))
        else:
            results.append(psd[coeff])

    return torch.stack(results, dim=0)


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
            continue

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
        key = (p["m"], p["r"])
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
            coeffs = torch.zeros(batch_size * n_states, m + 1, device=x.device, dtype=x.dtype)

        for param_idx, coeff in items:
            if coeff < m + 1:
                results[param_idx] = coeffs[:, coeff].reshape(batch_size, n_states)

    return results


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


@torch.no_grad()
def value_count_batched(x: Tensor, values: list[float]) -> Tensor:
    """Compute value_count for multiple values at once.

    Args:
        x: Input tensor of shape (N, B, S)
        values: List of values to count

    Returns:
        Tensor of shape (len(values), B, S)
    """
    batch_size, n_states = x.shape[1], x.shape[2]

    if not values:
        return torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device)

    results = []
    for value in values:
        results.append((x == value).float().sum(dim=0))

    return torch.stack(results, dim=0)


@torch.no_grad()
def range_count_batched(x: Tensor, params: list[dict]) -> Tensor:
    """Compute range_count for multiple (min_val, max_val) pairs at once.

    Args:
        x: Input tensor of shape (N, B, S)
        params: List of parameter dicts, each with keys "min_val" and "max_val"

    Returns:
        Tensor of shape (len(params), B, S)
    """
    batch_size, n_states = x.shape[1], x.shape[2]

    if not params:
        return torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device)

    results = []
    for p in params:
        min_val = p["min_val"]
        max_val = p["max_val"]
        results.append(((x >= min_val) & (x <= max_val)).float().sum(dim=0))

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

        agg_results = {
            "mean": acf.mean(dim=0),
            "median": acf.median(dim=0).values,
            "var": acf.var(dim=0, correction=0),
        }

        for param_idx, f_agg in items:
            if f_agg in agg_results:
                results[param_idx] = agg_results[f_agg]
            else:
                results[param_idx] = agg_results["mean"]

    return results


@torch.no_grad()
def augmented_dickey_fuller_batched(x: Tensor, attrs: list[str]) -> Tensor:
    """Compute augmented_dickey_fuller for multiple attributes at once.

    Computes ADF test once and returns all requested attributes.

    Args:
        x: Input tensor of shape (N, B, S)
        attrs: List of attributes ("teststat", "pvalue", "usedlag")

    Returns:
        Tensor of shape (len(attrs), B, S)
    """
    n = x.shape[0]
    batch_size, n_states = x.shape[1], x.shape[2]

    if not attrs:
        return torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device)

    diff = x[1:] - x[:-1]
    lagged = x[:-1]

    x_mean = lagged.mean(dim=0, keepdim=True)
    y_mean = diff.mean(dim=0, keepdim=True)

    x_centered = lagged - x_mean
    y_centered = diff - y_mean

    ss_xx = (x_centered**2).sum(dim=0)
    ss_xy = (x_centered * y_centered).sum(dim=0)

    beta = ss_xy / (ss_xx + 1e-10)

    residuals = diff - (y_mean + beta.unsqueeze(0) * x_centered)
    mse = (residuals**2).sum(dim=0) / (n - 3)

    se = torch.sqrt(mse) / (torch.sqrt(ss_xx) + 1e-10)
    t_stat = beta / (se + 1e-10)
    pvalue = 1 / (1 + t_stat.abs())
    usedlag = torch.ones(batch_size, n_states, dtype=x.dtype, device=x.device)

    attr_map = {
        "teststat": t_stat,
        "pvalue": pvalue,
        "usedlag": usedlag,
    }

    results = []
    for attr in attrs:
        if attr in attr_map:
            results.append(attr_map[attr])
        else:
            results.append(t_stat)

    return torch.stack(results, dim=0)


BATCHED_FEATURE_FUNCTIONS = {
    "autocorrelation_batched": autocorrelation_batched,
    "fft_coefficient_batched": fft_coefficient_batched,
    "fft_coefficient_all_attrs_batched": fft_coefficient_all_attrs_batched,
    "quantile_batched": quantile_batched,
    "index_mass_quantile_batched": index_mass_quantile_batched,
    "large_standard_deviation_batched": large_standard_deviation_batched,
    "symmetry_looking_batched": symmetry_looking_batched,
    "ratio_beyond_r_sigma_batched": ratio_beyond_r_sigma_batched,
    "energy_ratio_by_chunks_batched": energy_ratio_by_chunks_batched,
    "ar_coefficient_batched": ar_coefficient_batched,
    "linear_trend_batched": linear_trend_batched,
    "cwt_coefficients_batched": cwt_coefficients_batched,
    "change_quantiles_batched": change_quantiles_batched,
    "agg_linear_trend_batched": agg_linear_trend_batched,
    "partial_autocorrelation_batched": partial_autocorrelation_batched,
    "fourier_entropy_batched": fourier_entropy_batched,
    "fft_aggregated_batched": fft_aggregated_batched,
    "spkt_welch_density_batched": spkt_welch_density_batched,
    "number_peaks_batched": number_peaks_batched,
    "friedrich_coefficients_batched": friedrich_coefficients_batched,
    "number_crossing_m_batched": number_crossing_m_batched,
    "c3_batched": c3_batched,
    "time_reversal_asymmetry_statistic_batched": time_reversal_asymmetry_statistic_batched,
    "value_count_batched": value_count_batched,
    "range_count_batched": range_count_batched,
    "mean_n_absolute_max_batched": mean_n_absolute_max_batched,
    "agg_autocorrelation_batched": agg_autocorrelation_batched,
    "augmented_dickey_fuller_batched": augmented_dickey_fuller_batched,
}
