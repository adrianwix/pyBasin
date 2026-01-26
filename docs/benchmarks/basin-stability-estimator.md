# Basin Stability Estimator

## Interactive Flame Graph

View the profiling results in speedscope:

[Open in speedscope](https://www.speedscope.app/#profileURL=https%3A%2F%2Fraw.githubusercontent.com%2Fadrianwix%2FpyBasin%2Fmain%2Fbenchmarks%2Fprofiling%2Fprofile.speedscope.json&title=pyBasin%20Profiling){ .md-button .md-button--primary }

## Example Run

The pendulum case study with 10,000 initial conditions using pyBasin defaults:

```
BASIN STABILITY ESTIMATION COMPLETE
Total time: 17.3210s
Timing Breakdown:
  1. Sampling:             0.0629s  (  0.4%)
  2. Integration:         12.1686s  ( 70.3%)
  3. Solution/Amps:        0.0571s  (  0.3%)
  4. Features:             0.4379s  (  2.5%)
  5. Filtering:            0.0047s  (  0.0%)
  6. Classification:       4.5817s  ( 26.5%)
  7. BS Computation:       0.0073s  (  0.0%)
```

### Expensive Steps

The three most computationally expensive steps are:

1. **ODE Integration (~70%)** — Solving the differential equations for all initial conditions. Uses JAX/Diffrax by default with GPU acceleration.

2. **Classification (~26%)** — HDBSCAN clustering with auto-tuning enabled, followed by KMeans to assign noise points to the nearest cluster.

3. **Feature Extraction (~2.5%)** — Extracts time series features from trajectories. The default `TorchFeatureExtractor` uses these statistical features: `median`, `mean`, `standard_deviation`, `variance`, `root_mean_square`, `maximum`, `absolute_maximum`, `minimum`, `delta`, `log_delta`.

!!! note "Feature Complexity"
More complex features (e.g., entropy, autocorrelation, frequency domain) can significantly increase extraction time. The default minimal set is chosen for speed while maintaining classification accuracy.

## Profiling Setup

The profile was generated using [Austin](https://github.com/P403n1x87/austin), a frame stack sampler for CPython.

The pendulum case study is run using pyBasin defaults—only the ODE system and area of interest (sampler bounds) are defined. All other components (solver, feature extractor, predictor) use their default configurations.

To generate a new profile:

```bash
./scripts/generate_profiling.sh
```

This runs the pendulum case study and outputs `profile.speedscope.json` for visualization in speedscope.
