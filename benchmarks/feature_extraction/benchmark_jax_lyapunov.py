# pyright: basic
"""General overview benchmark for JAX feature extractors vs nolds.

Compares JAX implementations of lyap_r, lyap_e, and corr_dim against nolds
baseline on logistic map and pendulum systems. Useful for validating correctness
and getting initial performance metrics. For detailed performance analysis, see
benchmark_jax_lyap_e.py (parallelization strategies) and benchmark_jax_lyap_r.py
(fitting methods comparison).
"""

import time

import jax
import jax.numpy as jnp
import nolds
import numpy as np

from case_studies.pendulum.pendulum_jax_ode import PendulumJaxODE
from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.feature_extractors.jax_corr_dim import corr_dim_batch
from pybasin.feature_extractors.jax_lyapunov_e import lyap_e_batch
from pybasin.feature_extractors.jax_lyapunov_r import lyap_r_batch
from pybasin.solvers import JaxSolver


def generate_pendulum_solutions(n_samples: int = 50, n_time: int = 500):
    """Generate pendulum trajectory solutions."""
    setup = setup_pendulum_system()
    ode: PendulumJaxODE = setup["ode_system"]  # type: ignore[assignment]
    solver: JaxSolver = setup["solver"]  # type: ignore[assignment]

    ics = setup["sampler"].sample(n_samples)
    print(f"Initial conditions shape: {ics.shape}")

    # Use the existing JaxSolver from setup
    _t_eval, y_values = solver.integrate(ode, ics)

    # y_values shape is (n_steps, batch, n_dims), which is what we need
    return jnp.array(y_values.cpu().numpy())


def generate_logistic_map_batch(n_samples: int = 100, n_time: int = 500):
    """Generate batch of logistic map trajectories."""

    def logistic_map(n, r=3.9, x0=0.1):
        x = [x0]
        for _ in range(n - 1):
            x.append(r * x[-1] * (1 - x[-1]))
        return np.array(x)

    np.random.seed(42)
    batch = [logistic_map(n_time, x0=0.1 + 0.01 * np.random.random()) for _ in range(n_samples)]
    # Shape (N_time, N_samples, 1)
    return jnp.array(batch).T[:, :, None]


def benchmark_jax_vs_nolds(solutions: jnp.ndarray, n_compare: int = 5):
    """Benchmark JAX implementation vs nolds."""
    n_samples = solutions.shape[1]

    # Warmup JAX
    _ = lyap_r_batch(solutions, emb_dim=10, lag=1, trajectory_len=20)

    # JAX benchmark
    start = time.time()
    le_jax = lyap_r_batch(solutions, emb_dim=10, lag=1, trajectory_len=20)
    jax.block_until_ready(le_jax)
    jax_time = time.time() - start
    print(f"JAX time for {n_samples} trajectories: {jax_time:.3f}s")
    print(f"JAX Lyapunov shape: {le_jax.shape}")
    print(f"JAX Lyapunov (first {n_compare}):\n{le_jax[:n_compare]}")

    # nolds benchmark (only first n_compare for speed)
    solutions_np = np.array(solutions)
    start = time.time()
    le_nolds = []
    for i in range(n_compare):
        le_states = []
        for s in range(solutions_np.shape[2]):
            le = nolds.lyap_r(
                solutions_np[:, i, s], emb_dim=10, lag=1, trajectory_len=20, fit="poly"
            )
            le_states.append(le)
        le_nolds.append(le_states)
    nolds_time = time.time() - start
    print(f"nolds time for {n_compare} trajectories: {nolds_time:.3f}s")
    print(f"nolds Lyapunov (first {n_compare}):\n{np.array(le_nolds)}")

    # Accuracy check
    max_diff = np.max(np.abs(np.array(le_jax[:n_compare]) - np.array(le_nolds)))
    print(f"Max difference: {max_diff:.2e}")

    # Speedup
    speedup = (nolds_time / n_compare * n_samples) / jax_time
    print(f"Speedup estimate: ~{speedup:.1f}x")

    return le_jax, le_nolds, jax_time, nolds_time


def benchmark_lyap_e_jax_vs_nolds(solutions: jnp.ndarray, n_compare: int = 5):
    """Benchmark JAX lyap_e implementation vs nolds."""
    n_samples = solutions.shape[1]
    emb_dim = 10
    matrix_dim = 4
    min_nb = min(2 * matrix_dim, matrix_dim + 4)

    # Warmup JAX
    _ = lyap_e_batch(solutions, emb_dim=emb_dim, matrix_dim=matrix_dim, min_nb=min_nb)

    # JAX benchmark
    start = time.time()
    le_jax = lyap_e_batch(solutions, emb_dim=emb_dim, matrix_dim=matrix_dim, min_nb=min_nb)
    jax.block_until_ready(le_jax)
    jax_time = time.time() - start
    print(f"JAX lyap_e time for {n_samples} trajectories: {jax_time:.3f}s")
    print(f"JAX lyap_e shape: {le_jax.shape}")
    print(f"JAX lyap_e (first {n_compare}):\n{le_jax[:n_compare]}")

    # nolds benchmark (only first n_compare for speed)
    solutions_np = np.array(solutions)
    start = time.time()
    le_nolds = []
    for i in range(n_compare):
        le_states = []
        for s in range(solutions_np.shape[2]):
            le = nolds.lyap_e(solutions_np[:, i, s], emb_dim=emb_dim, matrix_dim=matrix_dim)
            le_states.append(le)
        le_nolds.append(le_states)
    nolds_time = time.time() - start
    print(f"nolds lyap_e time for {n_compare} trajectories: {nolds_time:.3f}s")
    print(f"nolds lyap_e (first {n_compare}):\n{np.array(le_nolds)}")

    # Speedup
    speedup = (nolds_time / n_compare * n_samples) / jax_time
    print(f"Speedup estimate: ~{speedup:.1f}x")

    return le_jax, le_nolds, jax_time, nolds_time


def benchmark_corr_dim_jax_vs_nolds(solutions: jnp.ndarray, n_compare: int = 5):
    """Benchmark JAX corr_dim implementation vs nolds."""
    n_samples = solutions.shape[1]
    emb_dim = 4
    lag = 1

    # Warmup JAX
    _ = corr_dim_batch(solutions, emb_dim=emb_dim, lag=lag)

    # JAX benchmark
    start = time.time()
    cd_jax = corr_dim_batch(solutions, emb_dim=emb_dim, lag=lag)
    jax.block_until_ready(cd_jax)
    jax_time = time.time() - start
    print(f"JAX corr_dim time for {n_samples} trajectories: {jax_time:.3f}s")
    print(f"JAX corr_dim shape: {cd_jax.shape}")
    print(f"JAX corr_dim (first {n_compare}):\n{cd_jax[:n_compare]}")

    # nolds benchmark (only first n_compare for speed)
    solutions_np = np.array(solutions)
    start = time.time()
    cd_nolds = []
    for i in range(n_compare):
        cd_states = []
        for s in range(solutions_np.shape[2]):
            cd = nolds.corr_dim(solutions_np[:, i, s], emb_dim=emb_dim, lag=lag, fit="poly")
            cd_states.append(cd)
        cd_nolds.append(cd_states)
    nolds_time = time.time() - start
    print(f"nolds corr_dim time for {n_compare} trajectories: {nolds_time:.3f}s")
    print(f"nolds corr_dim (first {n_compare}):\n{np.array(cd_nolds)}")

    # Speedup
    speedup = (nolds_time / n_compare * n_samples) / jax_time
    print(f"Speedup estimate: ~{speedup:.1f}x")

    return cd_jax, cd_nolds, jax_time, nolds_time


if __name__ == "__main__":
    print("=" * 60)
    print("Benchmark: Logistic Map (simple 1D chaotic system)")
    print("=" * 60)
    logistic_solutions = generate_logistic_map_batch(n_samples=100, n_time=500)
    print(f"Solutions shape: {logistic_solutions.shape}")
    benchmark_jax_vs_nolds(logistic_solutions, n_compare=10)

    print("\n" + "=" * 60)
    print("Benchmark: Pendulum (2D ODE system)")
    print("=" * 60)
    try:
        pendulum_solutions = generate_pendulum_solutions(n_samples=50, n_time=500)
        print(f"Solutions shape: {pendulum_solutions.shape}")
        benchmark_jax_vs_nolds(pendulum_solutions, n_compare=5)
    except Exception as e:
        print(f"Pendulum benchmark failed: {e}")
        print("Skipping pendulum benchmark...")

    print("\n" + "=" * 60)
    print("Benchmark: lyap_e - Logistic Map")
    print("=" * 60)
    try:
        benchmark_lyap_e_jax_vs_nolds(logistic_solutions, n_compare=5)
    except Exception as e:
        print(f"lyap_e benchmark failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Benchmark: corr_dim - Logistic Map")
    print("=" * 60)
    try:
        benchmark_corr_dim_jax_vs_nolds(logistic_solutions, n_compare=10)
    except Exception as e:
        print(f"corr_dim benchmark failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Benchmark: corr_dim - Pendulum")
    print("=" * 60)
    try:
        benchmark_corr_dim_jax_vs_nolds(pendulum_solutions, n_compare=5)
    except Exception as e:
        print(f"corr_dim pendulum benchmark failed: {e}")
        import traceback

        traceback.print_exc()
