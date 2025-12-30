# pyright: basic
"""
GPU Profiling Benchmark for PyTorch Feature Extraction.

This script profiles GPU usage to understand the overhead breakdown:
- CPU dispatch time vs actual GPU kernel time
- Memory transfer overhead
- Top operations by GPU time

Usage:
    uv run python benchmarks/benchmark_torch_gpu_usage.py
    uv run python benchmarks/benchmark_torch_gpu_usage.py --batches=10000
    uv run python benchmarks/benchmark_torch_gpu_usage.py --top=30
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
from torch.profiler import ProfilerActivity, profile

from pybasin.ts_torch.settings import (
    TORCH_COMPREHENSIVE_FC_PARAMETERS,
    TORCH_GPU_FC_PARAMETERS,
)
from pybasin.ts_torch.torch_feature_processors import (
    count_features,
    extract_features_gpu,
    extract_features_gpu_batched,
)


def profile_gpu_extraction(
    n_timesteps: int = 200,
    n_batches: int = 1000,
    n_states: int = 1,
    use_comprehensive: bool = True,
    use_batched: bool = False,
    top_n: int = 20,
) -> dict:
    """Profile GPU feature extraction and return overhead analysis.

    Args:
        n_timesteps: Number of time steps per series
        n_batches: Number of batches (trajectories)
        n_states: Number of state variables
        use_comprehensive: Use full feature set (783 features) vs GPU-friendly subset
        use_batched: Use batched extraction vs standard
        top_n: Number of top operations to display

    Returns:
        Dictionary with profiling metrics
    """
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return {}

    # Select feature set
    fc_params = TORCH_COMPREHENSIVE_FC_PARAMETERS if use_comprehensive else TORCH_GPU_FC_PARAMETERS
    n_features = count_features(fc_params)
    extract_fn = extract_features_gpu_batched if use_batched else extract_features_gpu

    # Generate test data
    x = torch.randn(n_timesteps, n_batches, n_states).cuda()
    torch.cuda.synchronize()

    # Warmup
    print("Warming up...")
    _ = extract_fn(x, fc_params)
    torch.cuda.synchronize()

    # Profile
    print("Profiling...")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        _ = extract_fn(x, fc_params)
        torch.cuda.synchronize()

    # Get averages
    avgs = prof.key_averages()

    # Calculate totals
    cpu_total = avgs.self_cpu_time_total  # microseconds
    cuda_total = 0
    for e in avgs:
        cuda_total += e.self_device_time_total

    # Print results
    print("\n" + "=" * 100)
    print("GPU PROFILING RESULTS")
    print("=" * 100)

    print("\nConfiguration:")
    print(f"  Data shape: ({n_timesteps}, {n_batches}, {n_states})")
    print(f"  Total series: {n_batches * n_states}")
    print(f"  Features: {n_features}")
    print(f"  Extraction mode: {'batched' if use_batched else 'standard'}")

    print("\n" + "-" * 100)
    print(f"TOP {top_n} OPERATIONS BY CUDA TIME:")
    print("-" * 100)
    print(avgs.table(sort_by="self_cuda_time_total", row_limit=top_n))

    print("=" * 100)
    print("OVERHEAD ANALYSIS")
    print("=" * 100)

    cpu_ms = cpu_total / 1000
    cuda_ms = cuda_total / 1000
    overhead_ratio = cpu_total / cuda_total if cuda_total > 0 else float("inf")
    efficiency = 100 * cuda_total / cpu_total if cpu_total > 0 else 0

    print(f"\n  Total CPU time (dispatch + overhead): {cpu_ms:.1f}ms")
    print(f"  Total CUDA kernel time:               {cuda_ms:.1f}ms")
    print(f"  CPU overhead (non-GPU work):          {cpu_ms - cuda_ms:.1f}ms")
    print()
    print(f"  Overhead ratio: {overhead_ratio:.1f}x")
    print(f"    -> For every 1ms of GPU compute, there is {overhead_ratio:.1f}ms of CPU time")
    print()
    print(f"  GPU Efficiency: {efficiency:.1f}%")
    print(f"    -> Only {efficiency:.1f}% of wall time is actual GPU computation")
    print(f"    -> {100 - efficiency:.1f}% is Python/dispatch overhead")

    # Breakdown by category
    print("\n" + "-" * 100)
    print("BREAKDOWN BY OPERATION CATEGORY:")
    print("-" * 100)

    categories = {
        "Sorting (quantile/median)": ["sort", "argsort", "topk"],
        "FFT operations": ["fft", "rfft", "irfft"],
        "Convolution (CWT)": ["conv", "cudnn"],
        "Memory copies": ["copy_", "Memcpy"],
        "Reductions (sum/mean)": ["sum", "mean", "reduce"],
        "Elementwise ops": ["mul", "add", "sub", "div", "abs"],
    }

    for cat_name, keywords in categories.items():
        cat_cuda = 0
        cat_count = 0
        for e in avgs:
            if any(kw in e.key for kw in keywords):
                cat_cuda += e.self_device_time_total
                cat_count += e.count
        if cat_cuda > 0:
            pct = 100 * cat_cuda / cuda_total
            print(f"  {cat_name:30} {cat_cuda / 1000:8.2f}ms ({pct:5.1f}%) [{cat_count} calls]")

    print("\n" + "=" * 100)

    return {
        "cpu_time_ms": cpu_ms,
        "cuda_time_ms": cuda_ms,
        "overhead_ratio": overhead_ratio,
        "efficiency_pct": efficiency,
        "n_features": n_features,
        "n_batches": n_batches,
    }


def compare_standard_vs_batched(
    n_timesteps: int = 200,
    n_batches: int = 1000,
    n_states: int = 1,
):
    """Compare profiling between standard and batched GPU extraction."""
    print("\n" + "=" * 100)
    print("COMPARING STANDARD VS BATCHED GPU EXTRACTION")
    print("=" * 100)

    print("\n>>> STANDARD EXTRACTION <<<")
    standard = profile_gpu_extraction(
        n_timesteps=n_timesteps,
        n_batches=n_batches,
        n_states=n_states,
        use_batched=False,
        top_n=10,
    )

    print("\n>>> BATCHED EXTRACTION <<<")
    batched = profile_gpu_extraction(
        n_timesteps=n_timesteps,
        n_batches=n_batches,
        n_states=n_states,
        use_batched=True,
        top_n=10,
    )

    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY")
    print("=" * 100)
    print("                        Standard    Batched     Improvement")
    print(
        f"  CPU time:            {standard['cpu_time_ms']:8.1f}ms  {batched['cpu_time_ms']:8.1f}ms  {standard['cpu_time_ms'] / batched['cpu_time_ms']:.2f}x"
    )
    print(
        f"  CUDA time:           {standard['cuda_time_ms']:8.1f}ms  {batched['cuda_time_ms']:8.1f}ms  {standard['cuda_time_ms'] / batched['cuda_time_ms']:.2f}x"
    )
    print(
        f"  Efficiency:          {standard['efficiency_pct']:8.1f}%   {batched['efficiency_pct']:8.1f}%"
    )
    print("=" * 100)


def main():
    # Parse arguments
    n_batches = 1000
    top_n = 20
    compare = "--compare" in sys.argv

    for arg in sys.argv:
        if arg.startswith("--batches="):
            n_batches = int(arg.split("=")[1])
        if arg.startswith("--top="):
            top_n = int(arg.split("=")[1])

    print("=" * 100)
    print("GPU PROFILING BENCHMARK")
    print("=" * 100)

    if compare:
        compare_standard_vs_batched(n_batches=n_batches)
    else:
        profile_gpu_extraction(
            n_batches=n_batches,
            use_comprehensive=True,
            use_batched=False,
            top_n=top_n,
        )


if __name__ == "__main__":
    main()
