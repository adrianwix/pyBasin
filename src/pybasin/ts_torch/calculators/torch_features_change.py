# pyright: basic
import torch
from torch import Tensor
from torch.func import vmap

# =============================================================================
# CHANGE/DIFFERENCE BASED (5 features)
# =============================================================================


def _change_quantiles_core(
    x: Tensor,
    q_low: Tensor,
    q_high: Tensor,
    isabs: Tensor,
    is_var: Tensor,
) -> Tensor:
    """Core computation for change_quantiles (vmap-compatible).

    This is the inner computation that can be used with vmap.
    Quantiles must be pre-computed outside this function.

    Args:
        x: Input tensor (N, B, S)
        q_low: Low quantile values (B, S) or (1, B, S)
        q_high: High quantile values (B, S) or (1, B, S)
        isabs: Whether to use absolute differences (scalar bool tensor)
        is_var: Whether to compute variance instead of mean (scalar bool tensor)

    Returns:
        Result tensor of shape (B, S)
    """
    # Ensure quantiles have correct shape for broadcasting
    if q_low.dim() == 2:
        q_low = q_low.unsqueeze(0)
    if q_high.dim() == 2:
        q_high = q_high.unsqueeze(0)

    mask = (x >= q_low) & (x <= q_high)  # (N, B, S)

    # Compute differences between consecutive values
    diff = x[1:] - x[:-1]  # (N-1, B, S)
    diff_abs = diff.abs()

    # Select diff or diff_abs based on isabs (use torch.where for vmap compatibility)
    d = torch.where(isabs, diff_abs, diff)

    # Mask for valid differences: both endpoints must be in corridor
    valid_changes = mask[:-1] & mask[1:]  # (N-1, B, S)

    # Apply mask and compute aggregation
    masked_diff = d * valid_changes.float()
    count = valid_changes.float().sum(dim=0)  # (B, S)

    # Compute mean
    mean_result = masked_diff.sum(dim=0) / (count + 1e-10)

    # Compute variance
    sq_diff = (d - mean_result.unsqueeze(0)) ** 2 * valid_changes.float()
    var_result = sq_diff.sum(dim=0) / (count + 1e-10)

    # Select mean or var based on is_var (use torch.where for vmap compatibility)
    result = torch.where(is_var, var_result, mean_result)

    # Zero out where no valid changes
    return torch.where(count > 0, result, torch.zeros_like(result))


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


@torch.no_grad()
def change_quantiles(
    x: Tensor, ql: float, qh: float, isabs: bool = True, f_agg: str = "mean"
) -> Tensor:
    """Statistics of changes within quantile corridor.

    Computes statistics of consecutive value changes where both values
    fall within the [ql, qh] quantile range.

    Args:
        x: Input tensor (N, B, S)
        ql: Lower quantile (0.0 to 1.0)
        qh: Upper quantile (0.0 to 1.0), must be > ql
        isabs: If True, use absolute differences
        f_agg: Aggregation function, "mean" or "var"

    Returns:
        Result tensor of shape (B, S)
    """
    q_low = torch.quantile(x, ql, dim=0, keepdim=True)  # (1, B, S)
    q_high = torch.quantile(x, qh, dim=0, keepdim=True)  # (1, B, S)

    isabs_t = torch.tensor(isabs, dtype=torch.bool, device=x.device)
    is_var_t = torch.tensor(f_agg == "var", dtype=torch.bool, device=x.device)

    return _change_quantiles_core(x, q_low, q_high, isabs_t, is_var_t)


# =============================================================================
# BATCHED CHANGE FEATURES
# =============================================================================


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
