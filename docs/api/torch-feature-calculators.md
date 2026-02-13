# Torch Feature Calculators

!!! warning "Experimental"
These feature calculators are experimental reimplementations of tsfresh in pure PyTorch. Individual implementations have not been deeply validated against tsfresh for correctness in all cases. Results are close but not identical to tsfresh.

All feature functions follow a consistent tensor shape convention:

- **Input:** `(N, B, S)` where `N` = timesteps, `B` = batch size, `S` = state variables
- **Output:** `(B, S)` for scalar features, or `(K, B, S)` for multi-valued features where `K` is the number of values

Features are computed along the time dimension (`dim=0`), preserving batch and state dimensions. Functions suffixed with `_batched` compute several parameter variations in a single pass and return shape `(K, B, S)`.

---

## Statistical

::: pybasin.ts_torch.calculators.torch_features_statistical
options:
show_root_heading: false
heading_level: 3

---

## Change / Difference

::: pybasin.ts_torch.calculators.torch_features_change
options:
show_root_heading: false
heading_level: 3

---

## Counting

::: pybasin.ts_torch.calculators.torch_features_count
options:
show_root_heading: false
heading_level: 3

---

## Boolean

::: pybasin.ts_torch.calculators.torch_features_boolean
options:
show_root_heading: false
heading_level: 3

---

## Location

::: pybasin.ts_torch.calculators.torch_features_location
options:
show_root_heading: false
heading_level: 3

---

## Pattern / Streak

::: pybasin.ts_torch.calculators.torch_features_pattern
options:
show_root_heading: false
heading_level: 3

---

## Autocorrelation

::: pybasin.ts_torch.calculators.torch_features_autocorrelation
options:
show_root_heading: false
heading_level: 3

---

## Entropy / Complexity

::: pybasin.ts_torch.calculators.torch_features_entropy_complexity
options:
show_root_heading: false
heading_level: 3

---

## Frequency Domain

::: pybasin.ts_torch.calculators.torch_features_frequency
options:
show_root_heading: false
heading_level: 3

---

## Trend / Regression

::: pybasin.ts_torch.calculators.torch_features_trend
options:
show_root_heading: false
heading_level: 3

---

## Reoccurrence

::: pybasin.ts_torch.calculators.torch_features_reocurrance
options:
show_root_heading: false
heading_level: 3

---

## Advanced

::: pybasin.ts_torch.calculators.torch_features_advanced
options:
show_root_heading: false
heading_level: 3

---

## Dynamical Systems

::: pybasin.ts_torch.calculators.torch_features_dynamical
options:
show_root_heading: false
heading_level: 3
