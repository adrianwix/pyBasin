# Benchmarks Overview

This section documents pyBasin's performance characteristics and compares it against the original MATLAB implementation ([bSTAB-M](https://github.com/TUHH-DYN/bSTAB)).

## Test Hardware

- **CPU**: Intel Core Ultra 9 275HX
- **GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB VRAM)

## Key Findings

| Benchmark              | pyBasin vs MATLAB | Notes                          |
| ---------------------- | ----------------- | ------------------------------ |
| Solver (CPU, N=5k)     | ~1.2x faster      | JAX/Diffrax Dopri5             |
| Solver (GPU, N=100k)   | ~8.9x faster      | JAX/Diffrax near-constant time |
| End-to-End (GPU, 100k) | ~25x faster       | Full BS estimation pipeline    |

## Benchmark Pages

### [Basin Stability Estimator](basin-stability-estimator.md)

Detailed breakdown of the full estimation pipeline showing how time is distributed across each step (sampling, integration, feature extraction, classification). Includes an interactive flame graph for profiling analysis.

### [End-to-End Performance](end-to-end.md)

Compares the complete basin stability estimation workflow between pyBasin and MATLAB bSTAB-M across different sample sizes. Demonstrates pyBasin's scalability advantage, especially on GPU.

### [Solver Comparison](solvers.md)

Evaluates different ODE solver backends (JAX/Diffrax, PyTorch/torchdiffeq, SciPy) across CPU and GPU. Shows how JAX achieves near-constant integration time on GPU regardless of sample size.

### [Feature Extraction](feature-extraction.md)

Compares feature extraction performance between pyBasin's PyTorch-based implementation and tsfresh. Analyzes the trade-offs between feature complexity and extraction speed.
