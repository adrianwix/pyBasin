# Plan: Implementing Missing Batched Features Using vmap

## Overview

This document outlines a plan for implementing batched versions of PyTorch feature calculators using `torch.vmap`. The goal is to improve GPU performance by reducing Python loop overhead when computing features with multiple parameter combinations.

## Current Status

### Already Implemented (12 features)
| Feature                  | Batched Function                   | Params |
| ------------------------ | ---------------------------------- | ------ |
| quantile                 | `quantile_batched`                 | 8      |
| large_standard_deviation | `large_standard_deviation_batched` | 19     |
| index_mass_quantile      | `index_mass_quantile_batched`      | 8      |
| autocorrelation          | `autocorrelation_batched`          | 10     |
| fft_coefficient          | `fft_coefficient_batched`          | 400    |
| cwt_coefficients         | `cwt_coefficients_batched`         | 60     |
| linear_trend             | `linear_trend_batched`             | 5      |
| ar_coefficient           | `ar_coefficient_batched`           | 11     |
| symmetry_looking         | `symmetry_looking_batched`         | 20     |
| energy_ratio_by_chunks   | `energy_ratio_by_chunks_batched`   | 10     |
| ratio_beyond_r_sigma     | `ratio_beyond_r_sigma_batched`     | 10     |

### Not Worth Batching (1-2 params)
| Feature                  | Params | Reason                   |
| ------------------------ | ------ | ------------------------ |
| binned_entropy           | 1      | Single param, no benefit |
| max_langevin_fixed_point | 1      | Single param, no benefit |
| number_cwt_peaks         | 2      | Only 2 params            |
| cid_ce                   | 2      | Only 2 params            |

### Excluded from GPU
| Feature             | Params | Reason                                   |
| ------------------- | ------ | ---------------------------------------- |
| permutation_entropy | 5      | Uses Python loops, extremely slow on GPU |

## Implementation Priority

### High Priority (>10 params, significant speedup potential)

#### 1. `change_quantiles_batched` - 80 params ⭐⭐⭐
**Status:** Benchmark complete, vmap approach tested
**Speedup:** 8.7x with vmap vs baseline loop
**Implementation approach:**
- Create `_change_quantiles_core(x, q_low, q_high, isabs, is_var)` that takes pre-computed quantiles
- Pre-compute all unique quantiles outside vmap
- Use `torch.where` instead of Python if/else for `isabs` and `f_agg`
- vmap over `(q_low, q_high, isabs, is_var)` dimensions

#### 2. `agg_linear_trend_batched` - 48 params ⭐⭐⭐
**Implementation approach:**
- Group by `(chunk_size, f_agg)` combinations (3 chunk_sizes × 4 f_aggs = 12 groups)
- Within each group, compute all 4 attrs at once (slope, intercept, rvalue, stderr)
- Pre-compute chunked aggregations outside vmap
- Core function: `_agg_linear_trend_core(x_agg, attr_idx)` where `x_agg` is the aggregated chunks

#### 3. `partial_autocorrelation_batched` - 10 params ⭐⭐
**Implementation approach:**
- Similar to `autocorrelation_batched` but using Durbin-Levinson algorithm
- Compute all lags in one pass using matrix operations
- Return selected lag indices

### Medium Priority (3-5 params)

#### 4. `fourier_entropy_batched` - 5 params
**Implementation approach:**
- Compute FFT once
- vmap over different `bins` values
- Core function: `_fourier_entropy_core(fft_result, bins)`

#### 5. `lempel_ziv_complexity_batched` - 5 params
**Implementation approach:**
- This is tricky as LZ complexity uses sequential scanning
- May need to compute binned versions outside and vmap over bin counts
- Consider if vmap provides benefit here

#### 6. `linear_trend_timewise_batched` - 5 params
**Implementation approach:**
- Same as `linear_trend_batched` but with datetime-based x values
- Reuse `linear_trend_batched` logic with modified time axis

#### 7. `fft_aggregated_batched` - 4 params
**Implementation approach:**
- Compute FFT once
- Compute all 4 aggregations (centroid, variance, skew, kurtosis) at once
- Return as stacked tensor

#### 8. `friedrich_coefficients_batched` - 4 params
**Implementation approach:**
- Compute polynomial fit once (already fits degree m)
- Return all coefficients 0 to m in one call
- No vmap needed, just return multiple coefficients

#### 9. `number_peaks_batched` - 5 params
**Implementation approach:**
- This is challenging as peak detection is inherently sequential
- Consider computing for max `n` and then selecting subsets

#### 10. `spkt_welch_density_batched` - 3 params
**Implementation approach:**
- Compute Welch PSD once
- Extract multiple coefficient indices
- Similar to `fft_coefficient_batched`

### Low Priority (3 params each)

#### 11. `number_crossing_m_batched` - 3 params
```python
def _number_crossing_m_core(x, m):
    positive = x > m
    return (positive[:-1] != positive[1:]).float().sum(dim=0)
```
- vmap over `m` values

#### 12. `agg_autocorrelation_batched` - 3 params
- Compute full autocorrelation once
- Apply different aggregations (mean, median, var)
- Return all 3 at once

#### 13. `augmented_dickey_fuller_batched` - 3 params
- Compute ADF test once
- Return all 3 attrs (teststat, pvalue, usedlag)

#### 14. `time_reversal_asymmetry_statistic_batched` - 3 params
```python
def _time_reversal_asymmetry_core(x, lag):
    x_lag = x[lag:-lag]
    x_2lag = x[2*lag:]
    x_0 = x[:-2*lag]
    return (x_2lag**2 * x_lag - x_lag * x_0**2).mean(dim=0)
```
- vmap over `lag` values

#### 15. `c3_batched` - 3 params
```python
def _c3_core(x, lag):
    return (x[:-2*lag] * x[lag:-lag] * x[2*lag:]).mean(dim=0)
```
- vmap over `lag` values

#### 16. `value_count_batched` - 3 params
```python
def _value_count_core(x, value):
    return (x == value).float().sum(dim=0)
```
- vmap over `value` values

#### 17. `range_count_batched` - 3 params
```python
def _range_count_core(x, min_val, max_val):
    return ((x >= min_val) & (x <= max_val)).float().sum(dim=0)
```
- vmap over `(min_val, max_val)` pairs

#### 18. `mean_n_absolute_max_batched` - 3 params
```python
def _mean_n_absolute_max_core(x, n):
    top_vals, _ = x.abs().topk(n, dim=0)
    return top_vals.mean(dim=0)
```
- vmap over `n` values

## Implementation Pattern

The vmap approach follows this pattern:

```python
# 1. Create core function without torch.quantile or Python if/else
def _feature_core(x, param1, param2, ...):
    """Core computation, vmap-compatible."""
    # Use torch.where instead of if/else
    result = torch.where(condition, value_if_true, value_if_false)
    return result

# 2. Create public API that calls core
@torch.no_grad()
def feature(x, param1, param2, ...):
    """Public API with Python parameters."""
    # Convert Python params to tensors
    param1_t = torch.tensor(param1, device=x.device)
    return _feature_core(x, param1_t, ...)

# 3. Create batched version using vmap
@torch.no_grad()
def feature_batched(x, params_list):
    """Batched version using vmap."""
    # Pre-compute shared values (quantiles, FFT, etc.)
    ...
    
    # Build parameter tensors
    param1_arr = torch.tensor([p["param1"] for p in params_list], device=x.device)
    
    # vmap over parameter dimension
    vmapped_fn = vmap(_feature_core, in_dims=(None, 0, 0, ...))
    return vmapped_fn(x, param1_arr, ...)
```

## Integration with torch_feature_processors.py

After implementing batched functions:

1. Add to imports in `torch_feature_processors.py`:
```python
from pybasin.feature_extractors.torch_batched_calculators import (
    ...,
    change_quantiles_batched,
    agg_linear_trend_batched,
)
```

2. Add to `BATCHABLE_FEATURE_NAMES`:
```python
BATCHABLE_FEATURE_NAMES = {
    ...,
    "change_quantiles",
    "agg_linear_trend",
}
```

3. Add handling in `extract_features_gpu_batched`:
```python
elif feature_name == "change_quantiles":
    batch_results = change_quantiles_batched(x_gpu, group)
    for idx, fname in enumerate(names):
        results[fname] = batch_results[idx]
```

## Testing Strategy

1. **Correctness:** Compare batched output vs baseline loop output
   - `assert (batched_result - loop_result).abs().max() < 1e-6`

2. **Performance:** Benchmark speedup vs baseline
   - Target: >2x speedup for features with >5 params

3. **GPU Memory:** Ensure batched versions don't OOM
   - Test with large batch sizes (10000+)

## Estimated Impact

| Feature                 | Params | Current Calls | After Batching | Estimated Speedup |
| ----------------------- | ------ | ------------- | -------------- | ----------------- |
| change_quantiles        | 80     | 80            | 1              | 5-10x             |
| agg_linear_trend        | 48     | 48            | ~12            | 3-5x              |
| partial_autocorrelation | 10     | 10            | 1              | 3-5x              |
| All others combined     | ~35    | 35            | ~10            | 2-3x              |

**Total estimated kernel launches reduced:** ~170 → ~25 (85% reduction)
