# Benchmarks Overview

## Purpose

Compare pyBasin performance against MATLAB bSTAB-M and evaluate different solver and feature extraction backends.

## Test Hardware

- **CPU**: Intel Core Ultra 9 275HX
- **GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB VRAM)

## Key Findings

| Benchmark              | pyBasin vs MATLAB | Notes                          |
| ---------------------- | ----------------- | ------------------------------ |
| Solver (CPU, N=5k)     | ~1.2x faster      | JAX/Diffrax Dopri5             |
| Solver (GPU, N=100k)   | ~8.9x faster      | JAX/Diffrax near-constant time |
| End-to-End (GPU, 100k) | ~25x faster       | Full BS estimation pipeline    |

## Detailed Results

- [End-to-End Performance](end-to-end.md)
- [Solver Comparison](solvers.md)
- [Feature Extraction](feature-extraction.md)
