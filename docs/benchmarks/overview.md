# Benchmarks Overview

!!! note "Documentation in Progress"
This page is under construction.

## Purpose

Compare pyBasin performance against MATLAB bSTAB-M and evaluate different solver and feature extraction backends.

## Test Hardware

- CPU: (specify)
- GPU: (specify)
- Memory: (specify)

## Key Findings

| Benchmark              | pyBasin vs MATLAB | Notes              |
| ---------------------- | ----------------- | ------------------ |
| Solver (CPU)           | ~10-15x faster    | Dopri5 methods     |
| Solver (GPU)           | ~10x faster       | At 100k samples    |
| Feature Extraction     | ~20x faster       | PyTorch vs tsfresh |
| End-to-End (GPU, 100k) | ~10x faster       | Full BS estimation |

## Detailed Results

- [End-to-End Performance](end-to-end.md)
- [Solver Comparison](solvers.md)
- [Feature Extraction](feature-extraction.md)
