# pyright: basic
"""Benchmark: vmap vs manual vectorization for change_quantiles_batched.

Compares different strategies for batching the change_quantiles feature:
1. Baseline loop - call the function 80 times
2. Manual vectorized - group by (ql, qh) pairs to share quantile computation
3. Fully vectorized - all params computed in parallel with broadcasting
4. vmap-based - use torch.vmap over pre-computed quantiles

Usage:
    uv run python benchmarks/benchmark_change_quantiles_batched.py
    uv run python benchmarks/benchmark_change_quantiles_batched.py --cpu
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
from torch import Tensor
from torch.func import vmap

from pybasin.feature_extractors.torch_feature_calculators import (
    _change_quantiles_core,
    change_quantiles,
)

# Check if GPU available
USE_CPU = "--cpu" in sys.argv
device = torch.device("cpu" if USE_CPU or not torch.cuda.is_available() else "cuda")
print(f"Device: {device}")

# Generate test data
n_timesteps = 500
n_batches = 1000
n_states = 2
x = torch.randn(n_timesteps, n_batches, n_states, device=device, dtype=torch.float32)

# The 80 parameter combinations from TORCH_COMPREHENSIVE_FC_PARAMETERS
params_list = [
    {"ql": ql, "qh": qh, "isabs": b, "f_agg": f}
    for ql in [0.0, 0.2, 0.4, 0.6, 0.8]
    for qh in [0.2, 0.4, 0.6, 0.8, 1.0]
    for b in [False, True]
    for f in ["mean", "var"]
    if ql < qh
]
print(f"Number of parameter combinations: {len(params_list)}")


# =============================================================================
# Baseline - call change_quantiles in a loop
# =============================================================================
def baseline_loop(x: Tensor, params_list: list[dict]) -> Tensor:
    """Call change_quantiles 60 times in a loop."""
    results = []
    for p in params_list:
        results.append(change_quantiles(x, **p))
    return torch.stack(results, dim=0)


# =============================================================================
# VMAP VERSION - use torch.vmap over pre-computed quantiles
# =============================================================================
# vmap uses _change_quantiles_core from torch_feature_calculators
# which is the same core logic as change_quantiles but takes pre-computed quantiles


@torch.no_grad()
def change_quantiles_batched_vmap(x: Tensor, params_list: list[dict]) -> Tensor:
    """Batched using vmap over pre-computed quantiles.

    Uses _change_quantiles_core which is the same logic as change_quantiles
    but accepts pre-computed quantiles for vmap compatibility.
    """
    # Compute all unique quantiles at once (outside vmap)
    unique_q_values = sorted({p["ql"] for p in params_list} | {p["qh"] for p in params_list})
    q_tensor = torch.tensor(unique_q_values, dtype=x.dtype, device=x.device)
    all_quantiles = torch.quantile(x, q_tensor, dim=0)  # (n_q, B, S)
    q_to_idx = {q: i for i, q in enumerate(unique_q_values)}

    # Build parameter tensors for vmap
    ql_indices = torch.tensor(
        [q_to_idx[p["ql"]] for p in params_list], dtype=torch.long, device=x.device
    )
    qh_indices = torch.tensor(
        [q_to_idx[p["qh"]] for p in params_list], dtype=torch.long, device=x.device
    )
    q_lows = all_quantiles[ql_indices]  # (P, B, S)
    q_highs = all_quantiles[qh_indices]  # (P, B, S)

    isabs_arr = torch.tensor([p["isabs"] for p in params_list], dtype=torch.bool, device=x.device)
    is_var_arr = torch.tensor(
        [p["f_agg"] == "var" for p in params_list], dtype=torch.bool, device=x.device
    )

    # Create vmapped function using _change_quantiles_core
    # vmap over dim 0 of q_lows, q_highs, isabs_arr, is_var_arr
    # x is broadcast (in_dims=None)
    vmapped_fn = vmap(_change_quantiles_core, in_dims=(None, 0, 0, 0, 0), out_dims=0)

    return vmapped_fn(x, q_lows, q_highs, isabs_arr, is_var_arr)


# =============================================================================
# MANUAL VECTORIZED VERSION - compute all at once
# =============================================================================
@torch.no_grad()
def change_quantiles_batched_manual(x: Tensor, params_list: list[dict]) -> Tensor:
    """Manually vectorized - groups by (ql, qh) pairs to share quantile computation."""
    _, batch_size, n_states = x.shape
    n_params = len(params_list)

    # Group params by (ql, qh) to avoid redundant quantile computations
    ql_qh_groups: dict[tuple[float, float], list[tuple[int, bool, str]]] = {}
    for idx, p in enumerate(params_list):
        key = (p["ql"], p["qh"])
        if key not in ql_qh_groups:
            ql_qh_groups[key] = []
        ql_qh_groups[key].append((idx, p["isabs"], p["f_agg"]))

    # Pre-compute all unique quantiles at once
    unique_q_values = sorted({p["ql"] for p in params_list} | {p["qh"] for p in params_list})
    q_tensor = torch.tensor(unique_q_values, dtype=x.dtype, device=x.device)
    all_quantiles = torch.quantile(x, q_tensor, dim=0)  # (n_q, B, S)
    q_to_idx = {q: i for i, q in enumerate(unique_q_values)}

    # Compute diff once (shared across all params)
    diff = x[1:] - x[:-1]
    diff_abs = diff.abs()

    results = torch.zeros(n_params, batch_size, n_states, dtype=x.dtype, device=x.device)

    for (ql, qh), variants in ql_qh_groups.items():
        q_low = all_quantiles[q_to_idx[ql]].unsqueeze(0)
        q_high = all_quantiles[q_to_idx[qh]].unsqueeze(0)
        mask = (x >= q_low) & (x <= q_high)
        valid_changes = mask[:-1] & mask[1:]
        count = valid_changes.float().sum(dim=0)

        for idx, isabs, f_agg in variants:
            d = diff_abs if isabs else diff
            masked_diff = d * valid_changes.float()

            if f_agg == "mean":
                res = masked_diff.sum(dim=0) / (count + 1e-10)
            else:
                mean_val = masked_diff.sum(dim=0) / (count + 1e-10)
                sq_diff = (d - mean_val.unsqueeze(0)) ** 2 * valid_changes.float()
                res = sq_diff.sum(dim=0) / (count + 1e-10)

            results[idx] = torch.where(count > 0, res, torch.zeros_like(res))

    return results


# =============================================================================
# FULLY VECTORIZED VERSION - no Python loops
# =============================================================================
@torch.no_grad()
def change_quantiles_batched_full_vectorized(x: Tensor, params_list: list[dict]) -> Tensor:
    """Fully vectorized using broadcasting - all params computed in parallel."""
    n, batch_size, n_states = x.shape
    n_params = len(params_list)

    # Extract param arrays
    qls = torch.tensor([p["ql"] for p in params_list], dtype=x.dtype, device=x.device)
    qhs = torch.tensor([p["qh"] for p in params_list], dtype=x.dtype, device=x.device)
    isabs_arr = torch.tensor([p["isabs"] for p in params_list], dtype=torch.bool, device=x.device)
    is_var_arr = torch.tensor(
        [p["f_agg"] == "var" for p in params_list], dtype=torch.bool, device=x.device
    )

    # Compute all needed quantiles at once
    unique_q = torch.unique(torch.cat([qls, qhs]))
    all_q = torch.quantile(x, unique_q, dim=0)

    # Build index mapping
    q_to_idx = {q.item(): i for i, q in enumerate(unique_q)}
    ql_indices = torch.tensor([q_to_idx[q.item()] for q in qls], dtype=torch.long, device=x.device)
    qh_indices = torch.tensor([q_to_idx[q.item()] for q in qhs], dtype=torch.long, device=x.device)

    # Get low/high quantiles for each param: (P, B, S)
    q_low = all_q[ql_indices]
    q_high = all_q[qh_indices]

    # Expand x for broadcasting: (N, 1, B, S)
    x_expanded = x.unsqueeze(1)

    # Mask: (N, P, B, S)
    mask = (x_expanded >= q_low.unsqueeze(0)) & (x_expanded <= q_high.unsqueeze(0))

    # Diff: (N-1, 1, B, S)
    diff = x[1:] - x[:-1]
    diff_expanded = diff.unsqueeze(1)

    # Apply isabs per param: (N-1, P, B, S)
    diff_vals = torch.where(
        isabs_arr.view(1, -1, 1, 1),
        diff_expanded.abs().expand(-1, n_params, -1, -1),
        diff_expanded.expand(-1, n_params, -1, -1),
    )

    # Valid changes: (N-1, P, B, S)
    valid_changes = mask[:-1] & mask[1:]

    # Masked diff
    masked_diff = diff_vals * valid_changes.float()
    count = valid_changes.float().sum(dim=0)

    # Mean result
    mean_result = masked_diff.sum(dim=0) / (count + 1e-10)

    # Var result
    sq_diff = (diff_vals - mean_result.unsqueeze(0)) ** 2 * valid_changes.float()
    var_result = sq_diff.sum(dim=0) / (count + 1e-10)

    # Select mean or var based on is_var_arr
    result = torch.where(is_var_arr.view(-1, 1, 1), var_result, mean_result)
    result = torch.where(count > 0, result, torch.zeros_like(result))

    return result


# =============================================================================
# BENCHMARK
# =============================================================================
def benchmark(name: str, func, *args, n_warmup: int = 3, n_runs: int = 10):
    """Benchmark a function."""
    # Warmup
    for _ in range(n_warmup):
        _ = func(*args)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time
    times = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = func(*args)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    mean_time = sum(times) / len(times) * 1000  # ms
    print(f"{name}: {mean_time:.2f} ms (result shape: {result.shape})")
    return result, mean_time


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("BENCHMARKING change_quantiles_batched implementations")
    print("=" * 70)
    print(f"Data shape: {x.shape}")
    print(f"Params: {len(params_list)} combinations")
    print()

    # Run benchmarks
    r1, t1 = benchmark("1. Baseline loop (60 calls)      ", baseline_loop, x, params_list)
    r2, t2 = benchmark(
        "2. Manual vectorized (grouped)   ", change_quantiles_batched_manual, x, params_list
    )
    r3, t3 = benchmark(
        "3. Fully vectorized (broadcast)  ",
        change_quantiles_batched_full_vectorized,
        x,
        params_list,
    )
    r4, t4 = benchmark(
        "4. vmap-based                    ", change_quantiles_batched_vmap, x, params_list
    )

    print("\n" + "=" * 70)
    print("SPEEDUPS vs baseline loop:")
    print("=" * 70)
    print(f"Manual vectorized:  {t1 / t2:.1f}x faster")
    print(f"Fully vectorized:   {t1 / t3:.1f}x faster")
    print(f"vmap-based:         {t1 / t4:.1f}x faster")

    # Verify correctness
    print("\n" + "=" * 70)
    print("CORRECTNESS CHECK (max abs diff from baseline):")
    print("=" * 70)
    print(f"Manual vectorized vs baseline: {(r1 - r2).abs().max().item():.2e}")
    print(f"Fully vectorized vs baseline:  {(r1 - r3).abs().max().item():.2e}")
    print(f"vmap-based vs baseline:        {(r1 - r4).abs().max().item():.2e}")
