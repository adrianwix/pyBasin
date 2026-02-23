# Zig Dopri5 Solver Benchmark

**Date:** 2026-02-23
**System:** Pendulum ODE (2D state space), Dormand-Prince 5(4) adaptive step-size
**Workload:** 50,000 initial conditions x 10,000 save points over t=[0, 1000]
**Output tensor:** shape (50000, 10000, 2) = 8.00 GB (float64)
**Build:** `zig build -Doptimize=ReleaseFast`
**Threads:** auto-detect (all available cores)

## Results

| Version                       | Time (s)  | Throughput (ICs/s) | Per-IC (us) |
| ----------------------------- | --------- | ------------------ | ----------- |
| Before (alloc + copy per IC)  | 2.462     | 20,309             | 49.2        |
| After (solve_into, zero-copy) | 2.342     | 21,351             | 46.8        |
| **Speedup**                   | **1.05x** |                    |             |

## What changed

### Zig side

- Added `solve_into()` to `Dopri5Generic`: writes results directly into a caller-provided
  contiguous `[*]f64` buffer instead of allocating `n_save` separate heap slices.
- `solve_batch` (C API) and per-thread workers now call `solve_into()` directly,
  each thread writing into its slice of the numpy-owned output array.
- The old `solve()` is preserved as a thin wrapper over `solve_into()` that creates
  a `[][]f64` view over a single contiguous allocation (no per-row heap allocs).
- Per-IC memory: previously O(n_save \* dim) heap allocations + element-wise copy;
  now zero result allocations, only 7 k-stage working buffers per thread (arena-reset).

### Python side

- Unified `solve()` and `solve_batch()` into a single `solve()` method.
- Accepts 1-D `(dim,)` or 2-D `(n_ics, dim)` initial conditions.
- Always returns `(t_eval, y)` with `y.shape = (n_ics, n_save, dim)`.
- Single-IC case is handled as a batch of size 1 routed through `solve_batch` C API.

## Notes

The 5% speedup is modest because the dominant cost is the RK45 integration itself
(7 ODE evaluations per accepted step x adaptive stepping over t=1000). The copy
overhead was already small relative to compute. The real wins are architectural:

1. No per-IC heap allocation for results (previously n_save separate allocs per IC).
2. No element-wise copy from scattered buffers to contiguous output.
3. Each pthread writes directly into its slice of the numpy array (cache-friendly).
4. Single Python API surface matches pybasin's `SolverProtocol` convention.
