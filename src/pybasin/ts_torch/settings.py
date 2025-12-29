from collections.abc import Callable, Mapping
from typing import Any

from torch import Tensor

from pybasin.ts_torch.calculators.torch_features_advanced import (
    benford_correlation,
    c3,
    energy_ratio_by_chunks,
    time_reversal_asymmetry_statistic,
)
from pybasin.ts_torch.calculators.torch_features_autocorrelation import (
    agg_autocorrelation,
    autocorrelation,
    partial_autocorrelation,
)
from pybasin.ts_torch.calculators.torch_features_boolean import (
    has_duplicate,
    has_duplicate_max,
    has_duplicate_min,
    has_large_standard_deviation,
    has_variance_larger_than_standard_deviation,
)
from pybasin.ts_torch.calculators.torch_features_change import (
    absolute_sum_of_changes,
    change_quantiles,
    mean_abs_change,
    mean_change,
    mean_second_derivative_central,
)
from pybasin.ts_torch.calculators.torch_features_count import (
    count_above,
    count_above_mean,
    count_below,
    count_below_mean,
    count_in_range,
    count_value,
)
from pybasin.ts_torch.calculators.torch_features_dynamical import (
    correlation_dimension,
    friedrich_coefficients,
    lyapunov_e,
    lyapunov_r,
    max_langevin_fixed_point,
)
from pybasin.ts_torch.calculators.torch_features_entropy_complexity import (
    approximate_entropy,
    binned_entropy,
    cid_ce,
    fourier_entropy,
    lempel_ziv_complexity,
    permutation_entropy,
    sample_entropy,
)
from pybasin.ts_torch.calculators.torch_features_frequency import (
    cwt_coefficients,
    fft_aggregated,
    fft_coefficient,
    spkt_welch_density,
)
from pybasin.ts_torch.calculators.torch_features_location import (
    first_location_of_maximum,
    first_location_of_minimum,
    index_mass_quantile,
    last_location_of_maximum,
    last_location_of_minimum,
)
from pybasin.ts_torch.calculators.torch_features_pattern import (
    longest_strike_above_mean,
    longest_strike_below_mean,
    number_crossing_m,
    number_cwt_peaks,
    number_peaks,
)
from pybasin.ts_torch.calculators.torch_features_reocurrance import (
    percentage_of_reoccurring_datapoints_to_all_datapoints,
    percentage_of_reoccurring_values_to_all_values,
    ratio_value_number_to_time_series_length,
    sum_of_reoccurring_data_points,
    sum_of_reoccurring_values,
)
from pybasin.ts_torch.calculators.torch_features_statistical import (
    abs_energy,
    absolute_maximum,
    delta,
    kurtosis,
    length,
    log_delta,
    maximum,
    mean,
    mean_n_absolute_max,
    median,
    minimum,
    quantile,
    ratio_beyond_r_sigma,
    root_mean_square,
    skewness,
    standard_deviation,
    sum_values,
    symmetry_looking,
    variance,
    variation_coefficient,
)
from pybasin.ts_torch.calculators.torch_features_trend import (
    agg_linear_trend,
    ar_coefficient,
    augmented_dickey_fuller,
    linear_trend,
    linear_trend_timewise,
)

ALL_FEATURE_FUNCTIONS: dict[str, Callable[..., Tensor]] = {
    # Minimal features (10)
    "sum_values": sum_values,
    "median": median,
    "mean": mean,
    "length": length,
    "standard_deviation": standard_deviation,
    "variance": variance,
    "root_mean_square": root_mean_square,
    "maximum": maximum,
    "absolute_maximum": absolute_maximum,
    "minimum": minimum,
    # Custom (2)
    "delta": delta,
    "log_delta": log_delta,
    # Simple statistics (5)
    "abs_energy": abs_energy,
    "kurtosis": kurtosis,
    "skewness": skewness,
    "quantile": quantile,
    "variation_coefficient": variation_coefficient,
    # Change/difference (4)
    "absolute_sum_of_changes": absolute_sum_of_changes,
    "mean_abs_change": mean_abs_change,
    "mean_change": mean_change,
    "mean_second_derivative_central": mean_second_derivative_central,
    # Counting (4)
    "count_above": count_above,
    "count_above_mean": count_above_mean,
    "count_below": count_below,
    "count_below_mean": count_below_mean,
    # Boolean (5)
    "has_duplicate": has_duplicate,
    "has_duplicate_max": has_duplicate_max,
    "has_duplicate_min": has_duplicate_min,
    "has_variance_larger_than_standard_deviation": has_variance_larger_than_standard_deviation,
    "large_standard_deviation": has_large_standard_deviation,
    # Location (5)
    "first_location_of_maximum": first_location_of_maximum,
    "first_location_of_minimum": first_location_of_minimum,
    "last_location_of_maximum": last_location_of_maximum,
    "last_location_of_minimum": last_location_of_minimum,
    "index_mass_quantile": index_mass_quantile,
    # Streak/pattern (5)
    "longest_strike_above_mean": longest_strike_above_mean,
    "longest_strike_below_mean": longest_strike_below_mean,
    "number_crossing_m": number_crossing_m,
    "number_peaks": number_peaks,
    "number_cwt_peaks": number_cwt_peaks,
    # Autocorrelation (3)
    "autocorrelation": autocorrelation,
    "partial_autocorrelation": partial_autocorrelation,
    "agg_autocorrelation": agg_autocorrelation,
    # Entropy/complexity (7 - 2 not implemented: approximate_entropy, sample_entropy)
    "permutation_entropy": permutation_entropy,
    "binned_entropy": binned_entropy,
    "fourier_entropy": fourier_entropy,
    "lempel_ziv_complexity": lempel_ziv_complexity,
    "cid_ce": cid_ce,
    "approximate_entropy": approximate_entropy,
    "sample_entropy": sample_entropy,
    # Frequency domain (4)
    "fft_coefficient": fft_coefficient,
    "fft_aggregated": fft_aggregated,
    "spkt_welch_density": spkt_welch_density,
    "cwt_coefficients": cwt_coefficients,
    # Trend/regression (5)
    "linear_trend": linear_trend,
    "linear_trend_timewise": linear_trend_timewise,
    "agg_linear_trend": agg_linear_trend,
    "ar_coefficient": ar_coefficient,
    "augmented_dickey_fuller": augmented_dickey_fuller,
    # Reoccurrence (5)
    "percentage_of_reoccurring_datapoints_to_all_datapoints": percentage_of_reoccurring_datapoints_to_all_datapoints,
    "percentage_of_reoccurring_values_to_all_values": percentage_of_reoccurring_values_to_all_values,
    "sum_of_reoccurring_data_points": sum_of_reoccurring_data_points,
    "sum_of_reoccurring_values": sum_of_reoccurring_values,
    "ratio_value_number_to_time_series_length": ratio_value_number_to_time_series_length,
    # Advanced (4)
    "benford_correlation": benford_correlation,
    "c3": c3,
    "energy_ratio_by_chunks": energy_ratio_by_chunks,
    "time_reversal_asymmetry_statistic": time_reversal_asymmetry_statistic,
    # Remaining features (6)
    "change_quantiles": change_quantiles,
    "friedrich_coefficients": friedrich_coefficients,
    "max_langevin_fixed_point": max_langevin_fixed_point,
    "mean_n_absolute_max": mean_n_absolute_max,
    "ratio_beyond_r_sigma": ratio_beyond_r_sigma,
    "symmetry_looking": symmetry_looking,
    "count_in_range": count_in_range,
    "count_value": count_value,
    # Dynamical systems (3)
    "lyapunov_r": lyapunov_r,
    "lyapunov_e": lyapunov_e,
    "correlation_dimension": correlation_dimension,
}

FCParameters = Mapping[str, list[dict[str, Any]] | None]

# =============================================================================
# TSFRESH-COMPATIBLE CONFIGURATION (partial - will be extended)
# =============================================================================


TORCH_COMPREHENSIVE_FC_PARAMETERS: FCParameters = {
    # Minimal features (no parameters)
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
    # Simple statistics
    "abs_energy": None,
    "kurtosis": None,
    "skewness": None,
    "quantile": [{"q": q} for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]],
    "variation_coefficient": None,
    # Change/difference
    "absolute_sum_of_changes": None,
    "mean_abs_change": None,
    "mean_change": None,
    "mean_second_derivative_central": None,
    # Counting
    "count_above_mean": None,
    "count_below_mean": None,
    "count_above": [{"t": 0}],
    "count_below": [{"t": 0}],
    # Boolean
    "has_duplicate": None,
    "has_duplicate_max": None,
    "has_duplicate_min": None,
    "has_variance_larger_than_standard_deviation": None,
    "large_standard_deviation": [{"r": r * 0.05} for r in range(1, 20)],
    # Location
    "first_location_of_maximum": None,
    "first_location_of_minimum": None,
    "last_location_of_maximum": None,
    "last_location_of_minimum": None,
    "index_mass_quantile": [{"q": q} for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]],
    # Streak/pattern
    "longest_strike_above_mean": None,
    "longest_strike_below_mean": None,
    "number_crossing_m": [{"m": 0}, {"m": -1}, {"m": 1}],
    "number_peaks": [{"n": n} for n in [1, 3, 5, 10, 50]],
    "number_cwt_peaks": [{"max_width": n} for n in [1, 5]],
    # Autocorrelation
    "autocorrelation": [{"lag": lag} for lag in range(10)],
    "agg_autocorrelation": [{"f_agg": s, "maxlag": 40} for s in ["mean", "median", "var"]],
    "partial_autocorrelation": [{"lag": lag} for lag in range(10)],
    # Entropy/complexity
    "binned_entropy": [{"max_bins": 10}],
    "fourier_entropy": [{"bins": x} for x in [2, 3, 5, 10, 100]],
    "permutation_entropy": [{"tau": 1, "dimension": x} for x in [3, 4, 5, 6, 7]],
    "lempel_ziv_complexity": [{"bins": x} for x in [2, 3, 5, 10, 100]],
    "cid_ce": [{"normalize": True}, {"normalize": False}],
    # Frequency domain
    "fft_coefficient": [
        {"coeff": k, "attr": a} for a in ["real", "imag", "abs", "angle"] for k in range(100)
    ],
    "fft_aggregated": [{"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]],
    "spkt_welch_density": [{"coeff": coeff} for coeff in [2, 5, 8]],
    "cwt_coefficients": [
        {"widths": (w,), "coeff": coeff, "w": w} for w in [2, 5, 10, 20] for coeff in range(15)
    ],
    # Trend/regression
    "linear_trend": [
        {"attr": "pvalue"},
        {"attr": "rvalue"},
        {"attr": "intercept"},
        {"attr": "slope"},
        {"attr": "stderr"},
    ],
    "agg_linear_trend": [
        {"attr": attr, "chunk_size": i, "f_agg": f}
        for attr in ["rvalue", "intercept", "slope", "stderr"]
        for i in [5, 10, 50]
        for f in ["max", "min", "mean", "var"]
    ],
    "augmented_dickey_fuller": [
        {"attr": "teststat"},
        {"attr": "pvalue"},
        {"attr": "usedlag"},
    ],
    "ar_coefficient": [{"coeff": coeff, "k": 10} for coeff in range(11)],
    "linear_trend_timewise": [
        {"attr": "pvalue"},
        {"attr": "rvalue"},
        {"attr": "intercept"},
        {"attr": "slope"},
        {"attr": "stderr"},
    ],
    # Reoccurrence
    "percentage_of_reoccurring_datapoints_to_all_datapoints": None,
    "percentage_of_reoccurring_values_to_all_values": None,
    "sum_of_reoccurring_data_points": None,
    "sum_of_reoccurring_values": None,
    "ratio_value_number_to_time_series_length": None,
    # Advanced
    "benford_correlation": None,
    "time_reversal_asymmetry_statistic": [{"lag": lag} for lag in range(1, 4)],
    "c3": [{"lag": lag} for lag in range(1, 4)],
    "symmetry_looking": [{"r": r * 0.05} for r in range(20)],
    "change_quantiles": [
        {"ql": ql, "qh": qh, "isabs": b, "f_agg": f}
        for ql in [0.0, 0.2, 0.4, 0.6, 0.8]
        for qh in [0.2, 0.4, 0.6, 0.8, 1.0]
        for b in [False, True]
        for f in ["mean", "var"]
        if ql < qh
    ],
    "count_value": [{"value": value} for value in [0, 1, -1]],
    "count_in_range": [
        {"min_val": -1, "max_val": 1},
        {"min_val": -1e12, "max_val": 0},
        {"min_val": 0, "max_val": 1e12},
    ],
    "friedrich_coefficients": [{"coeff": coeff, "m": 3, "r": 30} for coeff in range(4)],
    "max_langevin_fixed_point": [{"m": 3, "r": 30}],
    "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": i} for i in range(10)],
    "ratio_beyond_r_sigma": [{"r": x} for x in [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10]],
    "mean_n_absolute_max": [{"number_of_maxima": n} for n in [3, 5, 7]],
    # Dynamical systems features
    "lyapunov_r": [
        {"emb_dim": 10, "lag": 1, "trajectory_len": 20, "tau": 1.0},
    ],
    "lyapunov_e": [
        {"emb_dim": 10, "matrix_dim": 4, "min_nb": 8, "min_tsep": 0, "tau": 1.0},
    ],
    "correlation_dimension": [
        {"emb_dim": 4, "lag": 1, "n_rvals": 50},
    ],
}

# =============================================================================
# GPU-FRIENDLY SUBSET (excludes features that are slower on GPU than CPU)
# =============================================================================
# Excluded features and why:
# - permutation_entropy: Uses Python loops, extremely slow on GPU (~1143ms)

GPU_EXCLUDED_FEATURES = {"permutation_entropy", "lempel_ziv_complexity"}

TORCH_GPU_FC_PARAMETERS: FCParameters = {
    k: v for k, v in TORCH_COMPREHENSIVE_FC_PARAMETERS.items() if k not in GPU_EXCLUDED_FEATURES
}

# Custom features (PyTorch-only, not in tsfresh)
TORCH_CUSTOM_FC_PARAMETERS: FCParameters = {
    "delta": None,
    "log_delta": None,
}

# Minimal feature names
MINIMAL_FEATURE_NAMES: list[str] = [
    "sum_values",
    "median",
    "mean",
    "length",
    "standard_deviation",
    "variance",
    "root_mean_square",
    "maximum",
    "absolute_maximum",
    "minimum",
    "delta",
    "log_delta",
]

# Minimal feature configuration (equivalent to tsfresh MinimalFCParameters + custom features)
TORCH_MINIMAL_FC_PARAMETERS: FCParameters = dict.fromkeys(MINIMAL_FEATURE_NAMES)

# Default configuration: minimal features + dynamical systems features
DEFAULT_TORCH_FC_PARAMETERS: FCParameters = {
    **TORCH_MINIMAL_FC_PARAMETERS,
    "lyapunov_r": [{"emb_dim": 10, "lag": 1, "trajectory_len": 20, "tau": 1.0}],
    "lyapunov_e": [{"emb_dim": 10, "matrix_dim": 4, "min_nb": 8, "min_tsep": 0, "tau": 1.0}],
    "correlation_dimension": [{"emb_dim": 4, "lag": 1, "n_rvals": 50}],
}
