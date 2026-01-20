# End-to-End Performance

!!! note "Documentation in Progress"
This page is under construction.

## Methodology

Compare full basin stability estimation times:

- Same ODE system (Pendulum)
- Same parameters (t_span, tolerances)
- Vary sample sizes: 10,000 / 50,000 / 100,000

## Implementations Compared

- MATLAB bSTAB-M (CPU)
- pyBasin + JaxSolver (CPU)
- pyBasin + JaxSolver (CUDA GPU)

## Results

| Samples | MATLAB CPU | pyBasin CPU | pyBasin GPU | GPU Speedup |
| ------- | ---------- | ----------- | ----------- | ----------- |
| 10,000  | ~11.3s     | ~9.3s       | ~11.7s      | 0.97x       |
| 100,000 | ~122s      | ~56s        | ~11.7s      | **10.4x**   |

## Key Finding

GPU time is nearly constant regardless of sample size due to parallelization.
