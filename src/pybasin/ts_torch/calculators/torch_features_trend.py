# pyright: basic
import torch
from torch import Tensor

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
# BATCHED TREND FEATURES
# =============================================================================


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
            results[param_idx] = trend_results[attr_to_idx[attr]]

    return results


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
