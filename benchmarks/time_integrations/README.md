# ODE Integration Benchmarks: MATLAB vs Python Solvers

This benchmark suite compares ODE integration performance across different numerical solvers for the **damped driven pendulum** system. The focus is purely on time integration performance, not on basin stability classification.

## System Description

**Damped Driven Pendulum ODE System:**
```
dy/dt = [y₁, -α·y₁ + T - K·sin(y₀)]
```

Where:
- `α = 0.1` (dissipation coefficient)
- `T = 0.5` (constant angular acceleration/torque)
- `K = 1.0` (stiffness coefficient, g/l)

**Integration Parameters:**
- Time span: [0, 1000]
- Relative tolerance: `1e-8`
- Absolute tolerance: `1e-6`
- Number of samples: 10,000 initial conditions
- Region of Interest: `[-2.618, 3.665] × [-10, 10]`
- Random seed: 42 (for reproducibility)

## Solvers Compared

### MATLAB
- **ode45**: Dormand-Prince 5(4) method with parallel processing (`parfor`)

### Python

#### Scipy (scipy.integrate.solve_ivp)
- **DOP853**: Dormand-Prince 8(5,3) method (primary)
- **RK45**: Runge-Kutta 4(5) method (alternative)

#### JAX + Diffrax
- **Dopri5**: Dormand-Prince 5(4) method (primary, equivalent to MATLAB ode45)
- **Tsit5**: Tsitouras 5(4) method (alternative)
- Supports CPU and GPU with vectorized integration via `jax.vmap`

#### Torchode
- **dopri5**: Dormand-Prince 5(4) method (primary)
- **tsit5**: Tsitouras 5(4) method (alternative)
- Supports CPU and GPU with batch integration

#### Torchdiffeq
- **dopri5**: Dormand-Prince 5(4) method (primary)
- **rk4**: Classic Runge-Kutta 4th order (alternative)
- Supports CPU and GPU with batch integration

## Solver Method Equivalence

| Solver             | Method                | Equivalent To                 |
| ------------------ | --------------------- | ----------------------------- |
| MATLAB ode45       | Dormand-Prince 5(4)   | **Baseline**                  |
| Scipy DOP853       | Dormand-Prince 8(5,3) | Higher order than baseline    |
| Scipy RK45         | Runge-Kutta 4(5)      | Similar accuracy to baseline  |
| JAX Diffrax Dopri5 | Dormand-Prince 5(4)   | **Exact equivalent**          |
| JAX Diffrax Tsit5  | Tsitouras 5(4)        | Similar order, more efficient |
| Torchode dopri5    | Dormant-Prince 5(4)   | **Exact equivalent**          |
| Torchode tsit5     | Tsitouras 5(4)        | Similar order, more efficient |
| Torchdiffeq dopri5 | Dormand-Prince 5(4)   | **Exact equivalent**          |
| Torchdiffeq rk4    | Classic RK4           | Lower order, fixed step       |

## Directory Structure

```
benchmarks/time_integrations/
├── README.md                          # This file
├── configs/
│   └── pendulum_params.json          # Shared configuration
├── matlab/
│   ├── ode_pendulum.m                # ODE system definition
│   └── benchmark_pendulum_ode45.m    # MATLAB benchmark script
├── python/
│   ├── benchmark_scipy.py            # Scipy benchmarks
│   ├── benchmark_jax_diffrax.py      # JAX + Diffrax benchmarks
│   ├── benchmark_torchode.py         # Torchode benchmarks
│   ├── benchmark_torchdiffeq.py      # Torchdiffeq benchmarks
│   └── compare_results.py            # Result comparison script
└── results/
    ├── all_timings.csv               # All timing results
    ├── comparison_summary.csv        # Summary comparison
    └── *.json                        # Individual result files
```

## Running the Benchmarks

### Prerequisites

#### MATLAB
- MATLAB R2020b or later
- Parallel Computing Toolbox (for `parfor`)

#### Python
All Python dependencies are in the main project `pyproject.toml`:
```bash
# Install dependencies
uv sync
```

Required packages:
- `numpy`, `scipy` (for Scipy benchmarks)
- `jax`, `jaxlib`, `diffrax` (for JAX benchmarks)
- `torch`, `torchode` (for Torchode benchmarks)
- `torch`, `torchdiffeq` (for Torchdiffeq benchmarks)
- `pandas` (for result comparison)

### Running MATLAB Benchmark

```bash
cd benchmarks/time_integrations/matlab
matlab -nodisplay -nosplash -r "benchmark_pendulum_ode45; exit"
```

Or run interactively in MATLAB:
```matlab
cd benchmarks/time_integrations/matlab
benchmark_pendulum_ode45
```

**Note**: Requires Parallel Computing Toolbox. The script automatically creates a parallel pool.

### Running Python Benchmarks

Each Python benchmark can be run independently:

```bash
cd benchmarks/time_integrations

# Scipy benchmarks (DOP853, RK45)
uv run python/benchmark_scipy.py

# JAX + Diffrax benchmarks (Dopri5, Tsit5, CPU + GPU)
uv run python/benchmark_jax_diffrax.py

# Torchode benchmarks (dopri5, tsit5, CPU + GPU)
uv run python/benchmark_torchode.py

# Torchdiffeq benchmarks (dopri5, rk4, CPU + GPU)
uv run python/benchmark_torchdiffeq.py
```

**GPU Support**: JAX, Torchode, and Torchdiffeq scripts automatically detect and use GPU if available. They fall back to CPU if no GPU is found.

### Comparing Results

After running benchmarks, compare results with:

```bash
cd benchmarks/time_integrations
uv run python/compare_results.py
```

This generates:
- Formatted comparison report (printed to console)
- `results/comparison_summary.csv` - Summary table
- `results/comparison_summary.json` - JSON format
- `results/comparison_full.csv` - All runs with metadata

### Verifying Basin Stability

To verify that ODE integrations are producing correct results, you can compute basin stability values:

```bash
cd benchmarks/time_integrations
uv run python/verify_basin_stability.py
```

This utility:
1. Loads trajectory data from benchmark results (`.npz` files)
2. Classifies each trajectory as Fixed Point (FP) or Limit Cycle (LC)
3. Computes basin stability fractions for each attractor
4. Compares with expected MATLAB reference values

**Classification Logic** (based on `PendulumFeatureExtractor`):
- Compute `delta = |max(omega) - mean(omega)|` in steady-state region
- If `delta < 0.01` → Fixed Point (FP)
- Otherwise → Limit Cycle (LC)

**Expected Results** (approximate, for α=0.1, T=0.5, K=1.0):
- FP basin stability: ~55-65%
- LC basin stability: ~35-45%

Results are saved to `results/*_basin_stability.json` files.

## Timeout Protection

All benchmarks include a **120-second timeout** to prevent extremely long processes:
- MATLAB: Serial mode checks timeout between integrations
- Python: All scripts check timeout between batches

If a timeout occurs, the benchmark saves partial results and reports how many integrations completed.

## Interpreting Results

### Key Metrics

1. **Elapsed Time (seconds)**: Total wall-clock time for all integrations
2. **Time per Integration (ms)**: Average time per initial condition
3. **Speedup**: Relative to MATLAB ode45 baseline (>1.0 is faster)
4. **Relative Performance (%)**: Percentage of baseline time (lower is better)

### Expected Performance Patterns

- **CPU Performance**: Scipy and JAX typically competitive with MATLAB
- **GPU Acceleration**: PyTorch-based solvers (Torchode, Torchdiffeq) show significant speedup on GPU
- **JAX vmap**: Excellent CPU vectorization, even better on GPU
- **Method Differences**: 
  - Dopri5/dopri5 methods should give nearly identical results
  - Tsit5/tsit5 often faster with similar accuracy
  - DOP853 (higher order) typically slower but more accurate
  - RK4 (fixed step) may be faster but less accurate

### Fair Comparisons

- **MATLAB ode45** vs **JAX Diffrax Dopri5**: Same algorithm, tests implementation
- **MATLAB ode45** vs **Torchode dopri5**: Same algorithm, tests framework overhead
- **CPU vs GPU**: Tests parallel efficiency and hardware utilization
- **Dopri5 vs Tsit5**: Tests algorithm efficiency at same order

## Notes on Reproducibility

1. **Deterministic Initial Conditions**: All benchmarks use fixed random seed (42)
2. **Same ODE System**: All implementations use identical parameters (α, T, K)
3. **Tolerances**: Consistent `rtol=1e-8`, `atol=1e-6` across all solvers
4. **Time Span**: Identical `[0, 1000]` integration interval
5. **Git Tracking**: Results include git commit hash for code version tracking

## Limitations & Caveats

1. **Parallel Strategies Differ**:
   - MATLAB: Multi-core CPU parallelism (`parfor`)
   - JAX: Vectorization (`vmap`) on CPU/GPU
   - PyTorch: Batch processing on CPU/GPU

2. **Hardware Dependencies**: GPU results depend heavily on GPU model and drivers

3. **JIT Compilation**: JAX and PyTorch benchmarks include warmup runs to exclude compilation time

4. **Solver Differences**: While methods are equivalent in order, implementations differ slightly

5. **No Basin Stability Classification**: This benchmark only measures ODE integration speed, not full basin stability pipeline

## References

- Menck, P. J., et al. "How basin stability complements the linear-stability paradigm." *Nature Physics* 9.2 (2013): 89-92.
- Dormand, J. R., and Prince, P. J. "A family of embedded Runge-Kutta formulae." *Journal of computational and applied mathematics* 6.1 (1980): 19-26.
- Tsitouras, Ch. "Runge–Kutta pairs of order 5(4) satisfying only the first column simplifying assumption." *Computers & Mathematics with Applications* 62.2 (2011): 770-775.

## Troubleshooting

### MATLAB Issues
- **No parallel pool**: Install Parallel Computing Toolbox or set `props.flagParallel = false` in script
- **Out of memory**: Reduce `n_samples` in `configs/pendulum_params.json`

### Python Issues
- **JAX GPU not detected**: Install `jax[cuda]` with correct CUDA version
- **PyTorch GPU not available**: Install PyTorch with CUDA support
- **Import errors**: Run `uv sync` to install all dependencies

### Timeout Issues
- **Benchmarks timing out**: Increase `max_integration_time_seconds` in config
- **Want longer runs**: Set to higher value or remove timeout checks

## Contributing

To add new solvers:
1. Create benchmark script in `python/` following existing patterns
2. Use shared config from `configs/pendulum_params.json`
3. Save results to `results/all_timings.csv` with same format
4. Document solver method and equivalence in this README

## License

See main project LICENSE file.
