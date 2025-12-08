# pyright: basic
"""
Benchmark to compare JAX vs tsfresh feature extraction timing.

This script times each feature in both JAX and tsfresh to compare performance.
It uses the same arguments for both implementations to ensure a fair comparison.

Feature Counts:
- MINIMAL_BATCH_FEATURES: 41 features (default, fast benchmarking)
- JAX_COMPREHENSIVE_FC_PARAMETERS (--all): 72 base features, 783 total with all permutations
- tsfresh ComprehensiveFCParameters (--comprehensive): 788 total feature calls
  (matrix_profile excluded due to missing stumpy dependency)

Usage:
    uv run python benchmarks/benchmark_jax_features_timing.py --individual-only
    uv run python benchmarks/benchmark_jax_features_timing.py --batch-only
    uv run python benchmarks/benchmark_jax_features_timing.py --batch-only --gpu
    uv run python benchmarks/benchmark_jax_features_timing.py --batch-only --batches=10000
    uv run python benchmarks/benchmark_jax_features_timing.py --batch-only --all  # Use all 783 JAX feature permutations
    uv run python benchmarks/benchmark_jax_features_timing.py --batch-only --comprehensive  # Use tsfresh ComprehensiveFCParameters (788 features)
"""

import os
import sys
from pathlib import Path
import warnings

# Suppress pandas FutureWarning from tsfresh
warnings.filterwarnings("ignore", category=FutureWarning, module="tsfresh")

# Parse --gpu flag BEFORE importing jax (env vars must be set first)
USE_GPU = "--gpu" in sys.argv
if not USE_GPU:
    os.environ["JAX_PLATFORMS"] = "cpu"

import time
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import feature_calculators as fc
from tsfresh.feature_extraction import ComprehensiveFCParameters

from pybasin.feature_extractors.jax_feature_calculators import (
    ALL_FEATURE_FUNCTIONS,
    JAX_COMPREHENSIVE_FC_PARAMETERS,
)

# Enable persistent compilation cache to speed up subsequent runs
cache_dir = Path(__file__).parent / "cache" / "jax_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(cache_dir))

# Parse --all flag
USE_ALL_FEATURES = "--all" in sys.argv

# Parse --comprehensive flag (use tsfresh ComprehensiveFCParameters)
USE_COMPREHENSIVE = "--comprehensive" in sys.argv

# =============================================================================
# MINIMAL FEATURE SET FOR FAST BATCH BENCHMARKS
# =============================================================================
# Subset of ~50 simpler/faster features (excludes slow features like CWT, AR, etc.)
# Format: {jax_name: kwargs} - None means no parameters
MINIMAL_BATCH_FEATURES: dict[str, dict | None] = {
    # MINIMAL_FEATURES (10)
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
    # SIMPLE_STATISTICS_FEATURES (4)
    "abs_energy": None,
    "kurtosis": None,
    "skewness": None,
    "variation_coefficient": None,
    # CHANGE_FEATURES (4)
    "absolute_sum_of_changes": None,
    "mean_abs_change": None,
    "mean_change": None,
    "mean_second_derivative_central": None,
    # COUNTING_FEATURES (2)
    "count_above_mean": None,
    "count_below_mean": None,
    # BOOLEAN_FEATURES (4)
    "has_duplicate": None,
    "has_duplicate_max": None,
    "has_duplicate_min": None,
    "has_variance_larger_than_standard_deviation": None,
    # LOCATION_FEATURES (4)
    "first_location_of_maximum": None,
    "first_location_of_minimum": None,
    "last_location_of_maximum": None,
    "last_location_of_minimum": None,
    # STREAK_FEATURES (2)
    "longest_strike_above_mean": None,
    "longest_strike_below_mean": None,
    # ENTROPY_FEATURES (2)
    "fourier_entropy": {"bins": 10},
    "cid_ce": {"normalize": True},
    # REOCCURRENCE_FEATURES (5)
    "percentage_of_reoccurring_datapoints_to_all_datapoints": None,
    "percentage_of_reoccurring_values_to_all_values": None,
    "sum_of_reoccurring_data_points": None,
    "sum_of_reoccurring_values": None,
    "ratio_value_number_to_time_series_length": None,
    # ADVANCED_FEATURES (4)
    "benford_correlation": None,
    "mean_n_absolute_max": {"number_of_maxima": 1},
    "ratio_beyond_r_sigma": {"r": 1.0},
    "symmetry_looking": {"r": 0.1},
}


# =============================================================================
# FEATURE CONFIGURATIONS FOR INDIVIDUAL BENCHMARKS
# =============================================================================
# Each entry: (jax_func_name, jax_kwargs, tsfresh_func_name, tsfresh_kwargs)
# We use ONE representative parameter set per feature (not all permutations)
# This gives ~70 individual feature comparisons

INDIVIDUAL_FEATURE_CONFIGS: list[tuple[str, dict, str, dict]] = [
    # === NO-PARAMETER FEATURES (36 features) ===
    ("sum_values", {}, "sum_values", {}),
    ("median", {}, "median", {}),
    ("mean", {}, "mean", {}),
    ("length", {}, "length", {}),
    ("standard_deviation", {}, "standard_deviation", {}),
    ("variance", {}, "variance", {}),
    ("root_mean_square", {}, "root_mean_square", {}),
    ("maximum", {}, "maximum", {}),
    ("absolute_maximum", {}, "absolute_maximum", {}),
    ("minimum", {}, "minimum", {}),
    ("abs_energy", {}, "abs_energy", {}),
    ("kurtosis", {}, "kurtosis", {}),
    ("skewness", {}, "skewness", {}),
    ("variation_coefficient", {}, "variation_coefficient", {}),
    ("absolute_sum_of_changes", {}, "absolute_sum_of_changes", {}),
    ("mean_abs_change", {}, "mean_abs_change", {}),
    ("mean_change", {}, "mean_change", {}),
    ("mean_second_derivative_central", {}, "mean_second_derivative_central", {}),
    ("count_above_mean", {}, "count_above_mean", {}),
    ("count_below_mean", {}, "count_below_mean", {}),
    ("has_duplicate", {}, "has_duplicate", {}),
    ("has_duplicate_max", {}, "has_duplicate_max", {}),
    ("has_duplicate_min", {}, "has_duplicate_min", {}),
    (
        "has_variance_larger_than_standard_deviation",
        {},
        "variance_larger_than_standard_deviation",
        {},
    ),
    ("first_location_of_maximum", {}, "first_location_of_maximum", {}),
    ("first_location_of_minimum", {}, "first_location_of_minimum", {}),
    ("last_location_of_maximum", {}, "last_location_of_maximum", {}),
    ("last_location_of_minimum", {}, "last_location_of_minimum", {}),
    ("longest_strike_above_mean", {}, "longest_strike_above_mean", {}),
    ("longest_strike_below_mean", {}, "longest_strike_below_mean", {}),
    (
        "percentage_of_reoccurring_datapoints_to_all_datapoints",
        {},
        "percentage_of_reoccurring_datapoints_to_all_datapoints",
        {},
    ),
    (
        "percentage_of_reoccurring_values_to_all_values",
        {},
        "percentage_of_reoccurring_values_to_all_values",
        {},
    ),
    ("sum_of_reoccurring_data_points", {}, "sum_of_reoccurring_data_points", {}),
    ("sum_of_reoccurring_values", {}, "sum_of_reoccurring_values", {}),
    (
        "ratio_value_number_to_time_series_length",
        {},
        "ratio_value_number_to_time_series_length",
        {},
    ),
    ("benford_correlation", {}, "benford_correlation", {}),
    # === PARAMETERIZED FEATURES (one config each) ===
    # Time reversal / lag features
    (
        "time_reversal_asymmetry_statistic",
        {"lag": 1},
        "time_reversal_asymmetry_statistic",
        {"lag": 1},
    ),
    ("c3", {"lag": 1}, "c3", {"lag": 1}),
    # Complexity
    ("cid_ce", {"normalize": True}, "cid_ce", {"normalize": True}),
    # Symmetry / distribution
    ("symmetry_looking", {"r": 0.1}, "symmetry_looking", {"param": [{"r": 0.1}]}),
    ("large_standard_deviation", {"r": 0.25}, "large_standard_deviation", {"r": 0.25}),
    ("quantile", {"q": 0.5}, "quantile", {"q": 0.5}),
    # Autocorrelation
    ("autocorrelation", {"lag": 1}, "autocorrelation", {"lag": 1}),
    (
        "agg_autocorrelation",
        {"f_agg": "mean", "maxlag": 40},
        "agg_autocorrelation",
        {"param": [{"f_agg": "mean", "maxlag": 40}]},
    ),
    ("partial_autocorrelation", {"lag": 1}, "partial_autocorrelation", {"param": [{"lag": 1}]}),
    # Peaks
    ("number_cwt_peaks", {"max_width": 5}, "number_cwt_peaks", {"n": 5}),
    ("number_peaks", {"n": 3}, "number_peaks", {"n": 3}),
    # Entropy
    ("binned_entropy", {"max_bins": 10}, "binned_entropy", {"max_bins": 10}),
    ("fourier_entropy", {"bins": 10}, "fourier_entropy", {"bins": 10}),
    (
        "permutation_entropy",
        {"tau": 1, "dimension": 3},
        "permutation_entropy",
        {"tau": 1, "dimension": 3},
    ),
    ("lempel_ziv_complexity", {"bins": 2}, "lempel_ziv_complexity", {"bins": 2}),
    # Index/mass
    ("index_mass_quantile", {"q": 0.5}, "index_mass_quantile", {"param": [{"q": 0.5}]}),
    # FFT / frequency
    (
        "fft_coefficient",
        {"coeff": 0, "attr": "abs"},
        "fft_coefficient",
        {"param": [{"coeff": 0, "attr": "abs"}]},
    ),
    (
        "fft_aggregated",
        {"aggtype": "centroid"},
        "fft_aggregated",
        {"param": [{"aggtype": "centroid"}]},
    ),
    ("spkt_welch_density", {"coeff": 2}, "spkt_welch_density", {"param": [{"coeff": 2}]}),
    (
        "cwt_coefficients",
        {"widths": (2,), "coeff": 0, "w": 2},
        "cwt_coefficients",
        {"param": [{"widths": (2,), "coeff": 0, "w": 2}]},
    ),
    # AR / trend
    ("ar_coefficient", {"coeff": 0, "k": 10}, "ar_coefficient", {"param": [{"coeff": 0, "k": 10}]}),
    ("linear_trend", {"attr": "slope"}, "linear_trend", {"param": [{"attr": "slope"}]}),
    (
        "linear_trend_timewise",
        {"attr": "slope"},
        "linear_trend_timewise",
        {"param": [{"attr": "slope"}]},
    ),
    (
        "agg_linear_trend",
        {"attr": "slope", "chunk_size": 10, "f_agg": "mean"},
        "agg_linear_trend",
        {"param": [{"attr": "slope", "chunk_len": 10, "f_agg": "mean"}]},
    ),
    (
        "augmented_dickey_fuller",
        {"attr": "teststat"},
        "augmented_dickey_fuller",
        {"param": [{"attr": "teststat"}]},
    ),
    # Change quantiles
    (
        "change_quantiles",
        {"ql": 0.0, "qh": 0.5, "isabs": False, "f_agg": "mean"},
        "change_quantiles",
        {"ql": 0.0, "qh": 0.5, "isabs": False, "f_agg": "mean"},
    ),
    # Counting with threshold
    ("count_above", {"t": 0}, "count_above", {"t": 0}),
    ("count_below", {"t": 0}, "count_below", {"t": 0}),
    ("number_crossing_m", {"m": 0}, "number_crossing_m", {"m": 0}),
    # Energy / ratio
    (
        "energy_ratio_by_chunks",
        {"num_segments": 10, "segment_focus": 0},
        "energy_ratio_by_chunks",
        {"param": [{"num_segments": 10, "segment_focus": 0}]},
    ),
    ("ratio_beyond_r_sigma", {"r": 1.0}, "ratio_beyond_r_sigma", {"r": 1.0}),
    # Value counting
    ("value_count", {"value": 0}, "value_count", {"value": 0}),
    ("range_count", {"min_val": -1, "max_val": 1}, "range_count", {"min": -1, "max": 1}),
    # Advanced/physics
    (
        "friedrich_coefficients",
        {"coeff": 0, "m": 3, "r": 30},
        "friedrich_coefficients",
        {"param": [{"coeff": 0, "m": 3, "r": 30}]},
    ),
    ("max_langevin_fixed_point", {"m": 3, "r": 30}, "max_langevin_fixed_point", {"m": 3, "r": 30}),
    (
        "mean_n_absolute_max",
        {"number_of_maxima": 3},
        "mean_n_absolute_max",
        {"number_of_maxima": 3},
    ),
]


def time_jax_feature(func, x_jax, kwargs, n_runs=3):
    """Time a JAX feature function."""
    try:
        # Warmup
        result = func(x_jax, **kwargs)
        jax.block_until_ready(result)

        # Time
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            result = func(x_jax, **kwargs)
            jax.block_until_ready(result)
            times.append(time.perf_counter() - t0)

        return np.mean(times), None
    except Exception as e:
        return None, str(e)


def time_tsfresh_feature(func_name, x_np, kwargs, n_batches, n_runs=3):
    """Time a tsfresh feature function over n_batches calls."""
    if not hasattr(fc, func_name):
        return None, f"Function {func_name} not found in tsfresh"

    tsfresh_fn = getattr(fc, func_name)

    try:
        # Test call
        if "param" in kwargs:
            # Generator-style function
            result = list(tsfresh_fn(x_np, kwargs["param"]))
        else:
            result = tsfresh_fn(x_np, **kwargs)

        # Time n_batches calls
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            for _ in range(n_batches):
                if "param" in kwargs:
                    _ = list(tsfresh_fn(x_np, kwargs["param"]))
                else:
                    _ = tsfresh_fn(x_np, **kwargs)
            times.append(time.perf_counter() - t0)

        return np.mean(times), None
    except Exception as e:
        return None, str(e)


def run_individual_benchmark(n_timesteps: int, n_batches: int, n_states: int, n_runs: int = 3):
    """Run individual feature benchmarks comparing JAX vs tsfresh."""
    print("\n" + "=" * 100)
    print("INDIVIDUAL FEATURE BENCHMARK")
    print("=" * 100)

    device = jax.devices()[0]
    total_series = n_batches * n_states

    print(f"\nDevice: {device}")
    print(
        f"Data: {n_timesteps} timesteps, {n_batches} batches x {n_states} states = {total_series} series"
    )
    print(
        f"JAX processes all {total_series} series at once, tsfresh runs {total_series}x per feature"
    )
    print(f"Timing: {n_runs} runs per feature\n")

    # Create test data
    np.random.seed(42)
    x_jax = jnp.array(np.random.randn(n_timesteps, n_batches, n_states))
    x_np = np.random.randn(n_timesteps)  # Single series for tsfresh

    print("-" * 100)
    print(f"{'Feature':<45} {'JAX':>12} {'tsfresh':>12} {'Speedup':>12} {'Status'}")
    print("-" * 100)

    results = []
    jax_total = 0.0
    tsfresh_total = 0.0
    n_success = 0
    n_jax_faster = 0
    n_tsfresh_faster = 0

    for jax_name, jax_kwargs, tsfresh_name, tsfresh_kwargs in INDIVIDUAL_FEATURE_CONFIGS:
        # Get JAX function
        if jax_name not in ALL_FEATURE_FUNCTIONS:
            print(f"  {jax_name:<45} {'SKIP':>12} {'':>12} {'':>12} JAX func not found")
            continue

        jax_func = ALL_FEATURE_FUNCTIONS[jax_name]

        # Time JAX
        jax_time, jax_err = time_jax_feature(jax_func, x_jax, jax_kwargs, n_runs)

        # Time tsfresh
        tsfresh_time, tsfresh_err = time_tsfresh_feature(
            tsfresh_name, x_np, tsfresh_kwargs, total_series, n_runs
        )

        # Format results
        if jax_time is not None:
            jax_str = f"{jax_time * 1000:8.2f}ms"
            jax_total += jax_time
        else:
            jax_str = "ERROR"

        if tsfresh_time is not None:
            tsfresh_str = f"{tsfresh_time * 1000:8.2f}ms"
            tsfresh_total += tsfresh_time
        else:
            tsfresh_str = "ERROR"

        if jax_time is not None and tsfresh_time is not None:
            speedup = tsfresh_time / jax_time
            if speedup >= 1:
                speedup_str = f"{speedup:8.1f}x"
                n_jax_faster += 1
            else:
                speedup_str = f"{1 / speedup:7.1f}x slower"
                n_tsfresh_faster += 1
            status = "OK"
            n_success += 1
        else:
            speedup_str = "N/A"
            status = jax_err or tsfresh_err or "ERROR"
            status = status[:20] if len(status) > 20 else status

        # Display name with params
        display_name = jax_name
        if jax_kwargs:
            param_str = ", ".join(f"{k}={v}" for k, v in list(jax_kwargs.items())[:2])
            display_name = f"{jax_name}({param_str})"
        display_name = display_name[:44]

        print(f"  {display_name:<45} {jax_str:>12} {tsfresh_str:>12} {speedup_str:>12} {status}")

        results.append(
            {
                "feature": jax_name,
                "jax_kwargs": jax_kwargs,
                "jax_time": jax_time,
                "tsfresh_time": tsfresh_time,
                "speedup": tsfresh_time / jax_time if jax_time and tsfresh_time else None,
            }
        )

    print("-" * 100)
    print(f"\n{'TOTALS':<45} {jax_total * 1000:8.2f}ms {tsfresh_total * 1000:8.2f}ms")

    # Summary
    print("\n" + "=" * 100)
    print("INDIVIDUAL BENCHMARK SUMMARY")
    print("=" * 100)
    print(f"  Features tested: {len(INDIVIDUAL_FEATURE_CONFIGS)}")
    print(f"  Successful comparisons: {n_success}")
    print(f"  JAX faster: {n_jax_faster}")
    print(f"  tsfresh faster: {n_tsfresh_faster}")
    if jax_total > 0:
        print(f"  Overall speedup: {tsfresh_total / jax_total:.1f}x")

    # Top 10 slowest JAX features
    print("\n" + "-" * 60)
    print("TOP 10 SLOWEST JAX FEATURES")
    print("-" * 60)
    sorted_by_jax = sorted(
        [r for r in results if r["jax_time"] is not None], key=lambda x: x["jax_time"], reverse=True
    )
    for i, r in enumerate(sorted_by_jax[:10], 1):
        pct = r["jax_time"] / jax_total * 100 if jax_total > 0 else 0
        print(f"  {i:2}. {r['feature']:<35} {r['jax_time'] * 1000:8.2f}ms ({pct:5.1f}%)")

    # Top 10 where tsfresh is faster
    print("\n" + "-" * 60)
    print("FEATURES WHERE TSFRESH IS FASTER")
    print("-" * 60)
    tsfresh_faster = [r for r in results if r["speedup"] is not None and r["speedup"] < 1]
    if tsfresh_faster:
        tsfresh_faster.sort(key=lambda x: x["speedup"])
        for i, r in enumerate(tsfresh_faster[:10], 1):
            print(f"  {i:2}. {r['feature']:<35} JAX {1 / r['speedup']:.1f}x slower")
    else:
        print("  None - JAX is faster for all features!")

    return results


def time_all_features_jax(x_jax, n_runs=3, parallel=False, use_jit_wrapper=False, use_all=False):
    """Time extracting all features at once using JAX.

    Args:
        x_jax: Input data
        n_runs: Number of timing runs
        parallel: If True, use ThreadPoolExecutor (CPU only)
        use_jit_wrapper: If True, wrap all feature extraction in a single JIT call (GPU optimal)
        use_all: If True, use all features from JAX_COMPREHENSIVE_FC_PARAMETERS

    Returns:
        Tuple of (warmup_time, mean_run_time, n_features)
    """
    feature_calls = []

    if use_all:
        # Use all features from JAX_COMPREHENSIVE_FC_PARAMETERS with ALL permutations
        for feature_name, param_list in JAX_COMPREHENSIVE_FC_PARAMETERS.items():
            if feature_name not in ALL_FEATURE_FUNCTIONS:
                continue
            func = ALL_FEATURE_FUNCTIONS[feature_name]
            if param_list is None:
                feature_calls.append((feature_name, func, {}))
            else:
                # Iterate ALL parameter permutations, not just the first one
                for params in param_list:
                    feature_calls.append((feature_name, func, params))
    else:
        # Use minimal feature subset for faster benchmarking
        for feature_name, kwargs in MINIMAL_BATCH_FEATURES.items():
            if feature_name not in ALL_FEATURE_FUNCTIONS:
                continue
            func = ALL_FEATURE_FUNCTIONS[feature_name]
            feature_calls.append((feature_name, func, kwargs or {}))

    def extract_all_sequential():
        results = {}
        for fname, func, kwargs in feature_calls:
            results[fname] = func(x_jax, **kwargs)
        jax.block_until_ready(list(results.values()))
        return results

    def extract_all_parallel():
        def run_feature(item):
            fname, func, kwargs = item
            result = func(x_jax, **kwargs)
            jax.block_until_ready(result)
            return fname, result

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = dict(executor.map(run_feature, feature_calls))
        return results

    @jax.jit
    def extract_all_jit(x):
        return {fname: func(x, **kwargs) for fname, func, kwargs in feature_calls}

    def extract_jit_wrapper():
        results = extract_all_jit(x_jax)
        jax.block_until_ready(list(results.values()))
        return results

    if use_jit_wrapper:
        extract_fn = extract_jit_wrapper
    elif parallel:
        extract_fn = extract_all_parallel
    else:
        extract_fn = extract_all_sequential

    # Warmup
    t0_warmup = time.perf_counter()
    _ = extract_fn()
    warmup_time = time.perf_counter() - t0_warmup

    # Time
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = extract_fn()
        times.append(time.perf_counter() - t0)

    return warmup_time, np.mean(times), len(feature_calls)


def time_all_features_tsfresh(
    n_timesteps: int,
    n_batches: int,
    use_all: bool,
    use_comprehensive: bool = False,
    n_runs: int = 3,
):
    """Time extracting features using tsfresh extract_features API."""
    # If comprehensive mode, use tsfresh's ComprehensiveFCParameters (~800 features)
    if use_comprehensive:
        fc_parameters = ComprehensiveFCParameters()
        n_features = len(fc_parameters)  # type: ignore[arg-type]
    else:
        fc_parameters = {}

    # Parameter name mappings from JAX to tsfresh
    param_remap = {
        "number_cwt_peaks": {"max_width": "n"},
        "agg_linear_trend": {"chunk_size": "chunk_len"},
        "range_count": {"min_val": "min", "max_val": "max"},
    }

    # JAX to tsfresh feature name mappings
    name_remap = {
        "has_variance_larger_than_standard_deviation": "variance_larger_than_standard_deviation",
    }

    if use_comprehensive:
        pass  # Already set fc_parameters above
    elif use_all:
        # Use all features from JAX_COMPREHENSIVE_FC_PARAMETERS
        source_features = JAX_COMPREHENSIVE_FC_PARAMETERS
        for feature_name, param_list in source_features.items():
            tsfresh_name = name_remap.get(feature_name, feature_name)

            if param_list is None:
                fc_parameters[tsfresh_name] = None
            else:
                params = param_list[0].copy()

                # Apply parameter name remapping
                if feature_name in param_remap:
                    for jax_key, tsfresh_key in param_remap[feature_name].items():
                        if jax_key in params:
                            params[tsfresh_key] = params.pop(jax_key)

                fc_parameters[tsfresh_name] = [params]
    else:
        # Use minimal feature subset
        for feature_name, kwargs in MINIMAL_BATCH_FEATURES.items():
            tsfresh_name = name_remap.get(feature_name, feature_name)

            if kwargs is None:
                fc_parameters[tsfresh_name] = None
            else:
                params = kwargs.copy()
                # Apply parameter name remapping
                if feature_name in param_remap:
                    for jax_key, tsfresh_key in param_remap[feature_name].items():
                        if jax_key in params:
                            params[tsfresh_key] = params.pop(jax_key)
                fc_parameters[tsfresh_name] = [params]

    if not use_comprehensive:
        n_features = len(fc_parameters)

    # Create DataFrame
    np.random.seed(42)
    data_rows = []
    for batch_id in range(n_batches):
        values = np.random.randn(n_timesteps)
        for t_idx, val in enumerate(values):
            data_rows.append({"id": batch_id, "time": t_idx, "value": val})

    df = pd.DataFrame(data_rows)

    # Time
    n_jobs = os.cpu_count() or 1
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = extract_features(
            df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=fc_parameters,
            disable_progressbar=True,
            n_jobs=n_jobs,
        )
        times.append(time.perf_counter() - t0)

    return np.mean(times), n_features


def run_batch_benchmark(
    n_timesteps: int,
    n_batches: int,
    n_states: int,
    use_gpu: bool,
    use_all: bool,
    use_comprehensive: bool = False,
):
    """Benchmark extracting all features at once (batch mode)."""
    print("\n" + "=" * 100)
    print("BATCH FEATURE EXTRACTION BENCHMARK")
    print("=" * 100)

    device = jax.devices()[0]
    total_series = n_batches * n_states

    if use_comprehensive:
        feature_set = "comprehensive (JAX 72 vs tsfresh ~800 features)"
    elif use_all:
        feature_set = "all (72 JAX features)"
    else:
        feature_set = "minimal (41 features)"

    print(f"\nDevice: {device}")
    print(
        f"Data: {n_timesteps} timesteps, {n_batches} batches x {n_states} states = {total_series} series"
    )
    print(f"Feature set: {feature_set}")
    print(f"tsfresh n_jobs={os.cpu_count()}")

    # Create test data
    np.random.seed(42)
    x_jax = jnp.array(np.random.randn(n_timesteps, n_batches, n_states))

    print("Timing tsfresh bulk extraction...")
    tsfresh_time, tsfresh_n_features = time_all_features_tsfresh(
        n_timesteps, total_series, use_all, use_comprehensive
    )

    if use_gpu:
        print("\nTiming JAX bulk extraction (parallel on GPU)...")
        jax_warmup, jax_time, jax_n_features = time_all_features_jax(
            x_jax, parallel=True, use_jit_wrapper=False, use_all=use_all or use_comprehensive
        )

        print("\n" + "-" * 80)
        print("BATCH EXTRACTION RESULTS (GPU MODE)")
        print("-" * 80)
        print(
            f"  JAX warmup ({jax_n_features} features):".ljust(45) + f"{jax_warmup * 1000:10.2f}ms"
        )
        print("  JAX post-warmup:".ljust(45) + f"{jax_time * 1000:10.2f}ms")
        print(
            f"  tsfresh ({tsfresh_n_features} features):".ljust(45)
            + f"{tsfresh_time * 1000:10.2f}ms"
        )
        print("-" * 80)

        if jax_time > 0:
            print(f"  {'Speedup (post-warmup):':45} {tsfresh_time / jax_time:10.1f}x")

        print(
            f"\n  Per-series: JAX {jax_time / total_series * 1e6:.2f}us vs tsfresh {tsfresh_time / total_series * 1e6:.2f}us"
        )

    else:
        print("\nTiming JAX bulk extraction (parallel)...")
        jax_warmup_par, jax_time_par, jax_n_features = time_all_features_jax(
            x_jax, parallel=True, use_all=use_all or use_comprehensive
        )

        print("\n" + "-" * 80)
        print("BATCH EXTRACTION RESULTS (CPU MODE)")
        print("-" * 80)
        print(
            f"  JAX parallel warmup ({jax_n_features} features):".ljust(45)
            + f"{jax_warmup_par * 1000:10.2f}ms"
        )
        print("  JAX parallel post-warmup:".ljust(45) + f"{jax_time_par * 1000:10.2f}ms")
        print(
            f"  tsfresh ({tsfresh_n_features} features):".ljust(45)
            + f"{tsfresh_time * 1000:10.2f}ms"
        )
        print("-" * 80)

        if jax_time_par > 0:
            print(f"  {'JAX parallel speedup:':45} {tsfresh_time / jax_time_par:10.1f}x")

        print("\n  Per-series timing:")
        print(f"    JAX parallel:   {jax_time_par / total_series * 1e6:.2f}us/series")
        print(f"    tsfresh:        {tsfresh_time / total_series * 1e6:.2f}us/series")


def main():
    print("=" * 100)
    print("JAX vs TSFRESH FEATURE TIMING BENCHMARK")
    print("=" * 100)

    # Parse arguments
    individual_only = "--individual-only" in sys.argv
    batch_only = "--batch-only" in sys.argv
    use_gpu = USE_GPU  # Already parsed at module level
    use_all = USE_ALL_FEATURES  # Already parsed at module level
    use_comprehensive = USE_COMPREHENSIVE  # Already parsed at module level

    # Parse --batches and --states
    n_batches = 1000
    n_states = 1
    for arg in sys.argv:
        if arg.startswith("--batches="):
            n_batches = int(arg.split("=")[1])
        if arg.startswith("--states="):
            n_states = int(arg.split("=")[1])

    n_timesteps = 200

    print("\nConfiguration:")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Batches: {n_batches}")
    print(f"  States: {n_states}")
    print(f"  Total series: {n_batches * n_states}")
    print(f"  GPU mode: {use_gpu}")
    print(f"  All features: {use_all}")
    print(f"  Comprehensive (tsfresh): {use_comprehensive}")
    print(f"  Individual benchmark: {not batch_only}")
    print(f"  Batch benchmark: {not individual_only}")

    if not batch_only:
        run_individual_benchmark(n_timesteps, n_batches, n_states)

    if not individual_only:
        run_batch_benchmark(n_timesteps, n_batches, n_states, use_gpu, use_all, use_comprehensive)

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
