# pyright: basic
"""PyTorch feature extraction processors for CPU and GPU.

This module provides reusable processors for extracting time series features
using PyTorch with different execution strategies:
- Sequential CPU extraction
- Parallel CPU extraction using multiprocessing
- GPU extraction using CUDA
- Batched GPU extraction (groups identical operations with different parameters)

These processors can be used for both benchmarking and production feature extraction.
"""

import multiprocessing as mp
import os
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from pybasin.feature_extractors.torch_batched_calculators import (
    agg_autocorrelation_batched,
    agg_linear_trend_batched,
    ar_coefficient_batched,
    augmented_dickey_fuller_batched,
    autocorrelation_batched,
    c3_batched,
    change_quantiles_batched,
    cwt_coefficients_batched,
    energy_ratio_by_chunks_batched,
    fft_aggregated_batched,
    fft_coefficient_batched,
    fourier_entropy_batched,
    friedrich_coefficients_batched,
    index_mass_quantile_batched,
    large_standard_deviation_batched,
    linear_trend_batched,
    mean_n_absolute_max_batched,
    number_crossing_m_batched,
    number_peaks_batched,
    partial_autocorrelation_batched,
    quantile_batched,
    range_count_batched,
    ratio_beyond_r_sigma_batched,
    spkt_welch_density_batched,
    symmetry_looking_batched,
    time_reversal_asymmetry_statistic_batched,
    value_count_batched,
)
from pybasin.feature_extractors.torch_feature_calculators import (
    ALL_FEATURE_FUNCTIONS,
    TORCH_COMPREHENSIVE_FC_PARAMETERS,
    TORCH_GPU_FC_PARAMETERS,
    FCParameters,
)

# Type alias for flexible feature parameters (supports both FCParameters and simpler dict formats)
FlexibleFCParameters = Mapping[str, list[dict[str, Any]] | dict[str, Any] | None]


def _normalize_fc_parameters(fc_params: FlexibleFCParameters) -> FCParameters:
    """Normalize flexible feature parameters to standard FCParameters format.

    Converts single-dict params to list format for consistency.

    Args:
        fc_params: Feature parameters in either format

    Returns:
        Normalized FCParameters with list format for all param values
    """
    result: dict[str, list[dict[str, Any]] | None] = {}
    for name, params in fc_params.items():
        if params is None:
            result[name] = None
        elif isinstance(params, list):
            result[name] = params
        else:
            # Single dict -> wrap in list
            result[name] = [params]
    return result


def _build_feature_calls(
    fc_parameters: FCParameters,
) -> list[tuple[str, Any, dict[str, Any]]]:
    """Build list of (feature_name, function, kwargs) tuples from parameters.

    Args:
        fc_parameters: Feature configuration mapping feature names to parameter lists

    Returns:
        List of (name, func, kwargs) tuples for each feature call
    """
    feature_calls: list[tuple[str, Any, dict[str, Any]]] = []

    for feature_name, param_list in fc_parameters.items():
        if feature_name not in ALL_FEATURE_FUNCTIONS:
            continue
        func = ALL_FEATURE_FUNCTIONS[feature_name]
        if param_list is None:
            feature_calls.append((feature_name, func, {}))
        else:
            for params in param_list:
                fname = f"{feature_name}__{_format_params(params)}"
                feature_calls.append((fname, func, dict(params)))

    return feature_calls


def _format_params(params: Mapping[str, Any]) -> str:
    """Format parameters into a string for feature naming."""
    return "__".join(f"{k}_{v}" for k, v in sorted(params.items()))


def _build_feature_names_and_kwargs(
    fc_parameters: FCParameters,
) -> list[tuple[str, dict[str, Any]]]:
    """Build list of (feature_name, kwargs) tuples for worker processes.

    Args:
        fc_parameters: Feature configuration mapping feature names to parameter lists

    Returns:
        List of (feature_name, kwargs) tuples
    """
    result: list[tuple[str, dict[str, Any]]] = []

    for feature_name, param_list in fc_parameters.items():
        if feature_name not in ALL_FEATURE_FUNCTIONS:
            continue
        if param_list is None:
            result.append((feature_name, {}))
        else:
            for params in param_list:
                result.append((feature_name, dict(params)))

    return result


def _process_chunk_worker(args: tuple[Any, list[tuple[str, dict[str, Any]]]]) -> dict[str, Any]:
    """Worker function for multiprocessing - must be at module level.

    Args:
        args: Tuple of (chunk_np, feature_names_and_kwargs)

    Returns:
        Dictionary mapping (feature_name, kwargs_str) to numpy results
    """
    chunk_np, feature_names_and_kwargs = args
    chunk = torch.from_numpy(chunk_np)
    torch.set_num_threads(1)

    results = {}
    for feature_name, kwargs in feature_names_and_kwargs:
        func = ALL_FEATURE_FUNCTIONS[feature_name]
        results[(feature_name, str(kwargs))] = func(chunk, **kwargs).numpy()
    return results


def extract_features_sequential(
    x: Tensor,
    fc_parameters: FlexibleFCParameters | None = None,
) -> dict[str, Tensor]:
    """Extract features sequentially on CPU.

    Args:
        x: Input tensor of shape (n_timesteps, n_batches, n_states)
        fc_parameters: Feature configuration. If None, uses TORCH_COMPREHENSIVE_FC_PARAMETERS

    Returns:
        Dictionary mapping feature names to result tensors of shape (n_batches, n_states)
    """
    if fc_parameters is None:
        fc_parameters = TORCH_COMPREHENSIVE_FC_PARAMETERS
    else:
        fc_parameters = _normalize_fc_parameters(fc_parameters)

    feature_calls = _build_feature_calls(fc_parameters)

    results: dict[str, Tensor] = {}
    for fname, func, kwargs in feature_calls:
        results[fname] = func(x, **kwargs)

    return results


def extract_features_parallel(
    x: Tensor,
    fc_parameters: FlexibleFCParameters | None = None,
    n_workers: int | None = None,
) -> dict[str, Tensor]:
    """Extract features in parallel using multiprocessing.

    Splits batches across worker processes for parallel CPU execution.

    Args:
        x: Input tensor of shape (n_timesteps, n_batches, n_states)
        fc_parameters: Feature configuration. If None, uses TORCH_COMPREHENSIVE_FC_PARAMETERS
        n_workers: Number of worker processes. If None, uses cpu_count

    Returns:
        Dictionary mapping feature names to result tensors of shape (n_batches, n_states)
    """
    if fc_parameters is None:
        fc_parameters = TORCH_COMPREHENSIVE_FC_PARAMETERS
    else:
        fc_parameters = _normalize_fc_parameters(fc_parameters)

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    feature_names_and_kwargs = _build_feature_names_and_kwargs(fc_parameters)

    # Split batches into chunks for parallel processing
    x_np = x.numpy()
    n_batches = x_np.shape[1]
    chunk_size = max(1, n_batches // n_workers)

    worker_args = []
    for i in range(0, n_batches, chunk_size):
        end = min(i + chunk_size, n_batches)
        chunk_np = x_np[:, i:end, :].copy()
        worker_args.append((chunk_np, feature_names_and_kwargs))

    # Run parallel extraction
    with mp.Pool(processes=n_workers) as pool:
        chunk_results = pool.map(_process_chunk_worker, worker_args)

    # Restore original thread count
    torch.set_num_threads(os.cpu_count() or 1)

    # Combine results from all chunks
    combined: dict[str, list[Any]] = {}
    for chunk_result in chunk_results:
        for key, value in chunk_result.items():
            if key not in combined:
                combined[key] = []
            combined[key].append(value)

    # Concatenate along batch dimension and convert back to tensors
    results: dict[str, Tensor] = {}
    for (feature_name, kwargs_str), arrays in combined.items():
        import numpy as np

        concatenated = np.concatenate(arrays, axis=0)
        # Build proper feature name
        if kwargs_str and kwargs_str != "{}":
            kwargs_dict = eval(kwargs_str)  # noqa: S307
            fname = f"{feature_name}__{_format_params(kwargs_dict)}"
        else:
            fname = feature_name
        results[fname] = torch.from_numpy(concatenated)

    return results


def extract_features_gpu(
    x: Tensor,
    fc_parameters: FlexibleFCParameters | None = None,
    use_gpu_friendly: bool = True,
) -> dict[str, Tensor]:
    """Extract features on GPU using CUDA.

    GPU naturally parallelizes across all batches without chunking.
    Data is moved to GPU and results are kept on GPU.

    Args:
        x: Input tensor of shape (n_timesteps, n_batches, n_states)
        fc_parameters: Feature configuration. If None, uses GPU-friendly or comprehensive set
        use_gpu_friendly: If True and fc_parameters is None, use GPU-optimized subset
                          that excludes slow features (quantile, permutation_entropy, etc.)

    Returns:
        Dictionary mapping feature names to result tensors on GPU

    Raises:
        RuntimeError: If CUDA is not available
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if fc_parameters is None:
        fc_parameters = (
            TORCH_GPU_FC_PARAMETERS if use_gpu_friendly else TORCH_COMPREHENSIVE_FC_PARAMETERS
        )
    else:
        fc_parameters = _normalize_fc_parameters(fc_parameters)

    device = torch.device("cuda")
    x_gpu = x.to(device)

    feature_calls = _build_feature_calls(fc_parameters)

    results: dict[str, Tensor] = {}
    for fname, func, kwargs in feature_calls:
        results[fname] = func(x_gpu, **kwargs)

    torch.cuda.synchronize()
    return results


def count_features(fc_parameters: FlexibleFCParameters | None = None) -> int:
    """Count the total number of features in a configuration.

    Args:
        fc_parameters: Feature configuration. If None, uses TORCH_COMPREHENSIVE_FC_PARAMETERS

    Returns:
        Total number of feature values that will be extracted
    """
    if fc_parameters is None:
        fc_parameters = TORCH_COMPREHENSIVE_FC_PARAMETERS
    else:
        fc_parameters = _normalize_fc_parameters(fc_parameters)

    return len(_build_feature_calls(fc_parameters))


# =============================================================================
# BATCHED GPU EXTRACTION - Groups identical operations with different parameters
# =============================================================================

# Features that can be batched - these have parameterized variants that can be computed together
BATCHABLE_FEATURE_NAMES = {
    "autocorrelation",
    "fft_coefficient",
    "quantile",
    "index_mass_quantile",
    "large_standard_deviation",
    "symmetry_looking",
    "ratio_beyond_r_sigma",
    "energy_ratio_by_chunks",
    "ar_coefficient",
    "linear_trend",
    "cwt_coefficients",
    "change_quantiles",
    "agg_linear_trend",
    "partial_autocorrelation",
    "fourier_entropy",
    "fft_aggregated",
    "spkt_welch_density",
    "number_peaks",
    "friedrich_coefficients",
    "number_crossing_m",
    "c3",
    "time_reversal_asymmetry_statistic",
    "value_count",
    "range_count",
    "mean_n_absolute_max",
    "agg_autocorrelation",
    "augmented_dickey_fuller",
}


def _group_batchable_features(
    fc_parameters: FCParameters,
) -> tuple[dict[str, list[tuple[str, dict[str, Any]]]], list[tuple[str, Any, dict[str, Any]]]]:
    """Group features into batchable groups and remaining individual calls.

    Returns:
        Tuple of (batched_groups, remaining_calls)
        - batched_groups: {feature_name: [(full_name, params), ...]}
        - remaining_calls: [(full_name, func, kwargs), ...]
    """
    batched_groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    remaining_calls: list[tuple[str, Any, dict[str, Any]]] = []

    for feature_name, param_list in fc_parameters.items():
        if feature_name not in ALL_FEATURE_FUNCTIONS:
            continue

        func = ALL_FEATURE_FUNCTIONS[feature_name]

        if feature_name in BATCHABLE_FEATURE_NAMES and param_list is not None:
            if feature_name not in batched_groups:
                batched_groups[feature_name] = []
            for params in param_list:
                fname = f"{feature_name}__{_format_params(params)}"
                batched_groups[feature_name].append((fname, dict(params)))
        elif param_list is None:
            remaining_calls.append((feature_name, func, {}))
        else:
            for params in param_list:
                fname = f"{feature_name}__{_format_params(params)}"
                remaining_calls.append((fname, func, dict(params)))

    return batched_groups, remaining_calls


def extract_features_gpu_batched(
    x: Tensor,
    fc_parameters: FlexibleFCParameters | None = None,
    use_gpu_friendly: bool = True,
) -> dict[str, Tensor]:
    """Extract features on GPU using batched operations for optimal performance.

    This implementation groups features with the same operation but different
    parameters (e.g., autocorrelation with lags 0-9) into single batched calls,
    significantly reducing kernel launch overhead.

    Key optimizations:
    - Batches fft_coefficient: 400 calls -> ~4 calls (one per attr type)
    - Batches autocorrelation: 10 calls -> 1 call
    - Batches cwt_coefficients: 60 calls -> 4 calls (one per width)
    - And more...

    Args:
        x: Input tensor of shape (n_timesteps, n_batches, n_states)
        fc_parameters: Feature configuration. If None, uses GPU-friendly or comprehensive set
        use_gpu_friendly: If True and fc_parameters is None, use GPU-optimized subset

    Returns:
        Dictionary mapping feature names to result tensors on GPU

    Raises:
        RuntimeError: If CUDA is not available
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    if fc_parameters is None:
        fc_parameters = (
            TORCH_GPU_FC_PARAMETERS if use_gpu_friendly else TORCH_COMPREHENSIVE_FC_PARAMETERS
        )
    else:
        fc_parameters = _normalize_fc_parameters(fc_parameters)

    device = torch.device("cuda")
    x_gpu = x.to(device)

    batched_groups, remaining_calls = _group_batchable_features(fc_parameters)

    results: dict[str, Tensor] = {}

    with torch.no_grad():
        for feature_name, group in batched_groups.items():
            if feature_name == "fft_coefficient":
                attrs_groups: dict[str, list[tuple[str, int]]] = {}
                for fname, params in group:
                    attr = params["attr"]
                    coeff = params["coeff"]
                    if attr not in attrs_groups:
                        attrs_groups[attr] = []
                    attrs_groups[attr].append((fname, coeff))

                for attr, items in attrs_groups.items():
                    names = [item[0] for item in items]
                    coeffs = [item[1] for item in items]
                    batch_results = fft_coefficient_batched(x_gpu, coeffs, attr)
                    for idx, fname in enumerate(names):
                        results[fname] = batch_results[idx]

            elif feature_name == "cwt_coefficients":
                names = [fname for fname, _ in group]
                param_dicts = [params for _, params in group]
                batch_results = cwt_coefficients_batched(x_gpu, param_dicts)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "ar_coefficient":
                k_groups: dict[int, list[tuple[str, int]]] = {}
                for fname, params in group:
                    k = params["k"]
                    coeff = params["coeff"]
                    if k not in k_groups:
                        k_groups[k] = []
                    k_groups[k].append((fname, coeff))

                for k, items in k_groups.items():
                    names = [item[0] for item in items]
                    coeffs = [item[1] for item in items]
                    batch_results = ar_coefficient_batched(x_gpu, k, coeffs)
                    for idx, fname in enumerate(names):
                        results[fname] = batch_results[idx]

            elif feature_name == "energy_ratio_by_chunks":
                ns_groups: dict[int, list[tuple[str, int]]] = {}
                for fname, params in group:
                    ns = params["num_segments"]
                    sf = params["segment_focus"]
                    if ns not in ns_groups:
                        ns_groups[ns] = []
                    ns_groups[ns].append((fname, sf))

                for ns, items in ns_groups.items():
                    names = [item[0] for item in items]
                    segment_focuses = [item[1] for item in items]
                    batch_results = energy_ratio_by_chunks_batched(x_gpu, ns, segment_focuses)
                    for idx, fname in enumerate(names):
                        results[fname] = batch_results[idx]

            elif feature_name == "linear_trend":
                names = [item[0] for item in group]
                attrs = [item[1]["attr"] for item in group]
                batch_results = linear_trend_batched(x_gpu, attrs)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "autocorrelation":
                names = [item[0] for item in group]
                lags = [item[1]["lag"] for item in group]
                batch_results = autocorrelation_batched(x_gpu, lags)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name in ("quantile", "index_mass_quantile"):
                names = [item[0] for item in group]
                qs = [item[1]["q"] for item in group]
                if feature_name == "quantile":
                    batch_results = quantile_batched(x_gpu, qs)
                else:
                    batch_results = index_mass_quantile_batched(x_gpu, qs)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name in (
                "large_standard_deviation",
                "symmetry_looking",
                "ratio_beyond_r_sigma",
            ):
                names = [item[0] for item in group]
                rs = [item[1]["r"] for item in group]
                if feature_name == "large_standard_deviation":
                    batch_results = large_standard_deviation_batched(x_gpu, rs)
                elif feature_name == "symmetry_looking":
                    batch_results = symmetry_looking_batched(x_gpu, rs)
                else:
                    batch_results = ratio_beyond_r_sigma_batched(x_gpu, rs)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "change_quantiles":
                names = [item[0] for item in group]
                param_dicts = [item[1] for item in group]
                batch_results = change_quantiles_batched(x_gpu, param_dicts)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "agg_linear_trend":
                names = [item[0] for item in group]
                param_dicts = [item[1] for item in group]
                batch_results = agg_linear_trend_batched(x_gpu, param_dicts)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "partial_autocorrelation":
                names = [item[0] for item in group]
                lags = [item[1]["lag"] for item in group]
                batch_results = partial_autocorrelation_batched(x_gpu, lags)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "fourier_entropy":
                names = [item[0] for item in group]
                bins_list = [item[1]["bins"] for item in group]
                batch_results = fourier_entropy_batched(x_gpu, bins_list)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "fft_aggregated":
                names = [item[0] for item in group]
                aggtypes = [item[1]["aggtype"] for item in group]
                batch_results = fft_aggregated_batched(x_gpu, aggtypes)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "spkt_welch_density":
                names = [item[0] for item in group]
                coeffs = [item[1]["coeff"] for item in group]
                batch_results = spkt_welch_density_batched(x_gpu, coeffs)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "number_peaks":
                names = [item[0] for item in group]
                ns = [item[1]["n"] for item in group]
                batch_results = number_peaks_batched(x_gpu, ns)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "friedrich_coefficients":
                names = [item[0] for item in group]
                param_dicts = [item[1] for item in group]
                batch_results = friedrich_coefficients_batched(x_gpu, param_dicts)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "number_crossing_m":
                names = [item[0] for item in group]
                ms = [item[1]["m"] for item in group]
                batch_results = number_crossing_m_batched(x_gpu, ms)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "c3":
                names = [item[0] for item in group]
                lags = [item[1]["lag"] for item in group]
                batch_results = c3_batched(x_gpu, lags)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "time_reversal_asymmetry_statistic":
                names = [item[0] for item in group]
                lags = [item[1]["lag"] for item in group]
                batch_results = time_reversal_asymmetry_statistic_batched(x_gpu, lags)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "value_count":
                names = [item[0] for item in group]
                values = [item[1]["value"] for item in group]
                batch_results = value_count_batched(x_gpu, values)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "range_count":
                names = [item[0] for item in group]
                param_dicts = [item[1] for item in group]
                batch_results = range_count_batched(x_gpu, param_dicts)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "mean_n_absolute_max":
                names = [item[0] for item in group]
                ns = [item[1]["number_of_maxima"] for item in group]
                batch_results = mean_n_absolute_max_batched(x_gpu, ns)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "agg_autocorrelation":
                names = [item[0] for item in group]
                param_dicts = [item[1] for item in group]
                batch_results = agg_autocorrelation_batched(x_gpu, param_dicts)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

            elif feature_name == "augmented_dickey_fuller":
                names = [item[0] for item in group]
                attrs = [item[1]["attr"] for item in group]
                batch_results = augmented_dickey_fuller_batched(x_gpu, attrs)
                for idx, fname in enumerate(names):
                    results[fname] = batch_results[idx]

        for fname, func, kwargs in remaining_calls:
            results[fname] = func(x_gpu, **kwargs)

    torch.cuda.synchronize()
    return results
