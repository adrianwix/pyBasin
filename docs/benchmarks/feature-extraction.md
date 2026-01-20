# Feature Extraction Benchmarks

!!! note "Documentation in Progress"
This page is under construction.

## Implementations Compared

- tsfresh (reference, CPU)
- TorchFeatureExtractor (CPU parallel)
- TorchFeatureExtractor (CUDA GPU)

## Results

| Backend | Mode       | Device | 10k batches time |
| ------- | ---------- | ------ | ---------------- |
| tsfresh | parallel   | cpu    | 34,465 ms        |
| PyTorch | parallel   | cpu    | 1,734 ms         |
| PyTorch | sequential | cpu    | 3,464 ms         |
| PyTorch | gpu        | cuda   | 7,702 ms         |

## Key Finding

PyTorch CPU parallel is **~20x faster** than tsfresh.

## Feature Accuracy

All features validated against tsfresh reference values.
