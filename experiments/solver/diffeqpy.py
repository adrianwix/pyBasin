# pyright: basic
"""Benchmark and correctness validation for the pendulum system using diffeqpy.

Uses diffeqpy (Julia's DifferentialEquations.jl via Python) with the DP5 solver
(Dormand-Prince 5th order) and EnsembleProblem for parallel trajectory integration.
The ODE is defined in Julia via ``de.seval``:
  - CPU: in-place form ``f(du, u, p, t)``
  - GPU: out-of-place form returning ``SVector{2}`` with ``ODEProblem{false}``
The ``prob_func`` is defined in Julia (via ``juliacall.Main.seval``) to avoid
Python↔Julia bridge overhead on every trajectory.

Supports two parallelization backends:
  - CPU: EnsembleThreads (multi-threaded)
  - GPU: EnsembleGPUKernel (CUDA) with StaticArrays via DiffEqGPU.jl

Reference: https://github.com/SciML/diffeqpy
           https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/

Scaling benchmark matches the structure of experiments/solver/heyoka.py.

Correctness is validated by computing basin stability and comparing to the JAX reference
pipeline defined in case_studies/pendulum/setup_pendulum_system.py.

Pendulum ODE:
    dθ/dt  = ω
    dω/dt  = −α·ω + T − K·sin(θ)

Parameters: alpha=0.1, T=0.5, K=1.0

Basin stability approximation (N=10000):
  - Random ICs sampled from the pendulum phase space
  - Feature: log_delta(ω) over the steady-state window [T_STEADY, T_SPAN[1]]
  - KNN (k=1) classifier trained on template trajectories for FP and LC attractors
  - Result: fraction of ICs converging to each attractor

Usage:
    uv run python -m experiments.solver.diffeqpy_bench
    uv run python -m experiments.solver.diffeqpy_bench --gpu  # GPU mode
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from diffeqpy import de  # type: ignore[import-untyped]
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------------------------
# Constants matching case_studies/pendulum/setup_pendulum_system.py
# ---------------------------------------------------------------------------
ALPHA: float = 0.1
T_PARAM: float = 0.5
K_PARAM: float = 1.0

T_SPAN: tuple[float, float] = (0.0, 1000.0)
T_STEADY: float = 950.0
N_STEADY_STEPS: int = 100

N_VALUES: list[int] = [100, 200, 500, 1000, 2000, 5000, 10_000, 20_000]
NUM_ROUNDS: int = 3
SEED: int = 42

TEMPLATE_Y0: list[list[float]] = [[0.4, 0.0], [2.7, 0.0]]
LABELS: list[str] = ["FP", "LC"]

RESULTS_DIR: Path = Path(__file__).parent / "results"

N_THREADS: int = os.cpu_count() or 1

USE_GPU: bool = False

_fast_prob: Any = None
_cuda_module: Any = None
_jl_prob_func: Any = None


# ---------------------------------------------------------------------------
# ODE (defined in Julia via de.seval for maximum speed)
# ---------------------------------------------------------------------------

# CPU: in-place ODE (evaluated in the DifferentialEquations module)
_jul_f: Any = de.seval("""
function f(du, u, p, t)
    theta, omega = u
    alpha, T_ext, K = p
    du[1] = omega
    du[2] = -alpha * omega + T_ext - K * sin(theta)
end
""")

_U0: list[float] = [0.0, 0.0]
_P: list[float] = [ALPHA, T_PARAM, K_PARAM]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _setup_cpu() -> None:
    """Build the base ODE problem for CPU ensemble solves."""
    global _fast_prob, _jl_prob_func
    from juliacall import Main as jl  # type: ignore[import-untyped]

    os.environ.setdefault("JULIA_NUM_THREADS", str(N_THREADS))
    _fast_prob = de.ODEProblem(_jul_f, _U0, T_SPAN, _P)

    jl.seval("using DifferentialEquations")
    _jl_prob_func = jl.seval("""
    (prob, i, repeat) -> remake(prob, u0 = jl_ics[i, :])
    """)


def _setup_gpu() -> None:
    """Build the GPU ODE problem with StaticArrays and Float32.

    Requires Julia packages (install once)::

        julia -e 'import Pkg; Pkg.add(["DifferentialEquations", "StaticArrays", "DiffEqGPU", "CUDA"])'
    """
    global _fast_prob, _cuda_module, _jl_prob_func
    from diffeqpy import cuda  # type: ignore[import-untyped]
    from juliacall import Main as jl  # type: ignore[import-untyped]

    _cuda_module = cuda

    jl.seval("using StaticArrays")
    jl.seval("""
    function f_gpu(u, p, t)
        theta = u[1]
        omega = u[2]
        alpha = p[1]
        T_ext = p[2]
        K = p[3]
        du1 = omega
        du2 = -alpha * omega + T_ext - K * sin(theta)
        return SVector{2}(du1, du2)
    end
    """)
    _fast_prob = jl.seval("""
    u0_gpu = @SVector Float32[0.0f0, 0.0f0]
    p_gpu = @SVector Float32[0.1f0, 0.5f0, 1.0f0]
    tspan_gpu = (0.0f0, 1000.0f0)
    ODEProblem{false}(f_gpu, u0_gpu, tspan_gpu, p_gpu)
    """)

    # GPU prob_func: convert ICs to SVector{2, Float32}
    _jl_prob_func = jl.seval("""
    (prob, i, repeat) -> begin
        u0 = @SVector Float32[Float32(jl_ics[i, 1]), Float32(jl_ics[i, 2])]
        remake(prob, u0 = u0)
    end
    """)


# ---------------------------------------------------------------------------
# Ensemble solving  (follows diffeqpy tutorial exactly)
# ---------------------------------------------------------------------------


def _solve_ensemble(y0_all: np.ndarray, saveat: list[float] | None = None) -> Any:
    """Solve the pendulum ODE for all ICs.

    :param y0_all: Initial conditions, shape (n, 2).
    :param saveat: Time points to save at. If ``None``, only the final state is kept.
    :return: EnsembleSolution object.
    """
    from juliacall import Main as jl  # type: ignore[import-untyped]

    n: int = len(y0_all)

    # Bulk-convert ICs to Julia Matrix{Float64} (single operation, no per-trajectory overhead)
    jl.jl_ics = jl.convert(jl.seval("Matrix{Float64}"), y0_all)

    ensemble_prob = de.EnsembleProblem(_fast_prob, prob_func=_jl_prob_func, safetycopy=False)

    save_kwargs: dict[str, Any] = (
        {"saveat": saveat} if saveat is not None else {"save_everystep": False, "save_start": False}
    )

    if USE_GPU:
        cuda = _cuda_module
        return de.solve(
            ensemble_prob,
            cuda.GPUTsit5(),
            cuda.EnsembleGPUKernel(cuda.CUDABackend()),
            trajectories=n,
            adaptive=False,
            dt=0.1,
            **save_kwargs,
        )
    return de.solve(
        ensemble_prob,
        de.DP5(),
        de.EnsembleThreads(),
        trajectories=n,
        **save_kwargs,
    )


# ---------------------------------------------------------------------------
# IC generation
# ---------------------------------------------------------------------------


def _generate_ics(n: int) -> np.ndarray:
    """Generate *n* random ICs inside the pendulum sampling region."""
    theta_eq: float = float(np.arcsin(T_PARAM / K_PARAM))
    rng = np.random.default_rng(SEED)
    return np.column_stack(
        [
            rng.uniform(-np.pi + theta_eq, np.pi + theta_eq, n),
            rng.uniform(-10.0, 10.0, n),
        ]
    )


# ---------------------------------------------------------------------------
# Scaling benchmark  (times only the ODE solve)
# ---------------------------------------------------------------------------


def _benchmark_scaling() -> list[dict[str, object]]:
    """Run the scaling benchmark. Only the ODE solve is timed."""
    print(f"\n{'=' * 60}")
    if USE_GPU:
        print("  diffeqpy EnsembleProblem (GPUTsit5, CUDA)")
    else:
        print(f"  diffeqpy EnsembleProblem (DP5, N_THREADS={N_THREADS})")
    print(f"{'=' * 60}")

    _gc = de.seval("GC.gc")

    warmup_y0: np.ndarray = _generate_ics(100)
    _solve_ensemble(warmup_y0)
    _solve_ensemble(warmup_y0)
    _gc()

    results: list[dict[str, object]] = []
    for n in N_VALUES:
        y0: np.ndarray = _generate_ics(n)
        round_times: list[float] = []

        for _ in range(NUM_ROUNDS):
            _gc()
            start: float = time.perf_counter()
            _solve_ensemble(y0)
            elapsed: float = time.perf_counter() - start
            round_times.append(elapsed)

        min_t: float = min(round_times)
        mean_t: float = mean(round_times)
        std_t: float = stdev(round_times) if len(round_times) > 1 else 0.0
        per_ic_us: float = min_t / n * 1e6

        print(
            f"  N={n:>7,}  min={min_t:.3f}s  mean={mean_t:.3f}s "
            f"+-{std_t:.3f}s  {per_ic_us:.1f} us/IC"
        )
        results.append(
            {
                "N": n,
                "round_times": round_times,
                "mean_time": mean_t,
                "std_time": std_t,
                "min_time": min_t,
                "max_time": max(round_times),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Feature extraction  (Python-side on data returned by diffeqpy)
# ---------------------------------------------------------------------------


def _log_delta(series: np.ndarray) -> float:
    """log(|max - mean| + 1e-10) over a 1-D time series."""
    return float(np.log(np.abs(series.max() - series.mean()) + 1e-10))


def _extract_features_batch(y0_all: np.ndarray) -> np.ndarray:
    """Return feature vector (log_delta of ω) for each IC in *y0_all*.

    ODE solve via diffeqpy with saveat; feature computation in numpy.
    """
    n: int = y0_all.shape[0]
    saveat: list[float] = list(np.linspace(T_STEADY, T_SPAN[1], N_STEADY_STEPS))
    n_t: int = len(saveat)

    sol = _solve_ensemble(y0_all, saveat=saveat)

    features: np.ndarray = np.empty(n)
    for i in range(n):
        traj = sol[i + 1]
        omega_series: np.ndarray = np.array([float(traj.u[j][1]) for j in range(n_t)])
        features[i] = _log_delta(omega_series)

    return features


# ---------------------------------------------------------------------------
# Basin stability
# ---------------------------------------------------------------------------

N_BASIN: int = 10000


def _compute_basin_stability() -> dict[str, object]:
    """Compute basin stability for the pendulum using diffeqpy."""
    print(f"\n{'=' * 60}")
    print(f"  Basin stability validation  (N={N_BASIN})")
    print(f"{'=' * 60}")

    template_y0: np.ndarray = np.array(TEMPLATE_Y0, dtype=float)
    template_features: np.ndarray = _extract_features_batch(template_y0).reshape(-1, 1)

    knn: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=1)
    knn.fit(template_features, LABELS)

    y0_all: np.ndarray = _generate_ics(N_BASIN)
    sample_features: np.ndarray = _extract_features_batch(y0_all).reshape(-1, 1)
    predictions: np.ndarray = np.array(knn.predict(sample_features))

    basin_stability: dict[str, float] = {
        label: float(np.mean(predictions == label)) for label in LABELS
    }

    print("  Basin stability estimates:")
    for label, bs in basin_stability.items():
        print(f"    {label}: {bs:.4f}")

    return {
        "n": N_BASIN,
        "basin_stability": basin_stability,
        "template_features": {
            LABELS[i]: float(template_features[i, 0]) for i in range(len(LABELS))
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    global USE_GPU

    parser = argparse.ArgumentParser(description="Benchmark diffeqpy ODE solver")
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU (CUDA) for parallel integration"
    )
    args = parser.parse_args()

    USE_GPU = args.gpu

    if USE_GPU:
        print("Initializing GPU support...")
        _setup_gpu()
        solver_name = "GPUTsit5"
        backend_info = "CUDA (EnsembleGPUKernel)"
    else:
        print("Initializing CPU support...")
        _setup_cpu()
        solver_name = "DP5"
        backend_info = f"CPU ({N_THREADS} threads, EnsembleThreads)"

    print(f"Backend: {backend_info}")
    print(f"Solver: {solver_name}")
    print(f"N values: {N_VALUES}")
    print(f"Rounds: {NUM_ROUNDS}  |  t_span: {T_SPAN}")

    benchmark_results: list[dict[str, object]] = _benchmark_scaling()
    basin_result: dict[str, object] = _compute_basin_stability()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    suffix: str = "_gpu" if USE_GPU else "_cpu"
    bench_path: Path = RESULTS_DIR / f"diffeqpy_scaling{suffix}.json"
    with open(bench_path, "w") as f:
        json.dump(
            {
                "solver": solver_name,
                "backend": "GPU" if USE_GPU else "CPU",
                "n_threads": N_THREADS if not USE_GPU else None,
                "num_rounds": NUM_ROUNDS,
                "t_span": list(T_SPAN),
                "benchmarks": benchmark_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {bench_path}")

    basin_path: Path = RESULTS_DIR / f"diffeqpy_basin_stability{suffix}.json"
    with open(basin_path, "w") as f:
        json.dump(basin_result, f, indent=2)
    print(f"Saved: {basin_path}")


if __name__ == "__main__":
    main()
