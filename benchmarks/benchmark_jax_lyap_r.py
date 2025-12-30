# pyright: basic
"""Deep dive benchmark for lyap_r (Rosenstein's method) fitting algorithms.

Compares 4 implementations with emphasis on fitting methods:
- nolds parallel RANSAC (robust to outliers, default sklearn behavior)
- nolds parallel poly (polynomial least squares fitting)
- JAX batch poly (vectorized polynomial least squares)
- JAX batch RANSAC (vectorized robust fitting)

Focuses on comparing RANSAC vs polynomial fitting for accuracy and performance
tradeoffs in batch Lyapunov exponent computation from trajectories.
"""

# Must set XLA_FLAGS BEFORE any JAX import - this must be at the very top
import os
import sys
from multiprocessing import cpu_count

_N_CPU_DEVICES = cpu_count()
_XLA_FLAG = f"--xla_force_host_platform_device_count={_N_CPU_DEVICES}"

if _XLA_FLAG not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = _XLA_FLAG
    # Re-exec the script with the new environment
    os.execv(sys.executable, [sys.executable] + sys.argv)

import logging
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

# Suppress pkg_resources deprecation warning from nolds
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import jax
import jax.numpy as jnp
import nolds
import numpy as np

from pybasin.feature_extractors.jax_lyapunov_r import lyap_r_batch

# Suppress other warnings
warnings.filterwarnings("ignore")
logging.getLogger("jax").setLevel(logging.CRITICAL)


def nolds_lyap_r_worker(args):
    """Worker function for parallel nolds lyap_r (RANSAC fit)."""
    data, emb_dim, lag, trajectory_len = args
    return nolds.lyap_r(data, emb_dim=emb_dim, lag=lag, trajectory_len=trajectory_len, fit="RANSAC")


def nolds_lyap_r_poly_worker(args):
    """Worker function for parallel nolds lyap_r (poly fit - same as JAX)."""
    data, emb_dim, lag, trajectory_len = args
    return nolds.lyap_r(data, emb_dim=emb_dim, lag=lag, trajectory_len=trajectory_len, fit="poly")


def generate_logistic_map_batch(n_samples=100, n_time=500):
    """Generate batch of logistic map trajectories."""

    def logistic_map(n, r=3.9, x0=0.1):
        x = [x0]
        for _ in range(n - 1):
            x.append(r * x[-1] * (1 - x[-1]))
        return np.array(x)

    np.random.seed(42)
    batch = [logistic_map(n_time, x0=0.1 + 0.01 * np.random.random()) for _ in range(n_samples)]
    return jnp.array(batch).T[:, :, None]


def run_benchmark(n_samples=100, n_workers=16):
    """Run the lyap_r benchmark suite.

    Args:
        n_samples: Number of trajectories to benchmark
        n_workers: Number of ProcessPool workers for parallel nolds
    """
    print(f"ProcessPool workers: {n_workers}")
    print(f"System CPU count: {cpu_count()}")

    print("=" * 70)
    print(f"BENCHMARK: lyap_r implementations ({n_samples} trajectories)")
    print("=" * 70)

    # Generate data
    solutions = generate_logistic_map_batch(n_samples=n_samples, n_time=500)
    solutions2 = solutions + 0.001  # Slightly different for timing
    solutions_np = np.array(solutions2)
    print(f"Solutions shape: {solutions.shape}")

    emb_dim, lag, trajectory_len = 10, 1, 20

    results = {}

    # 1. nolds parallel (RANSAC - default, robust fitting)
    print(f"\n[1/3] Running nolds parallel RANSAC ({n_workers} workers)...")
    args_list = [(solutions_np[:, i, 0], emb_dim, lag, trajectory_len) for i in range(n_samples)]
    start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        le_nolds_ransac = list(executor.map(nolds_lyap_r_worker, args_list))
    results["nolds_ransac"] = time.time() - start
    print(f"      Time: {results['nolds_ransac']:.3f}s")

    # 2. nolds parallel (poly - same algorithm as JAX)
    print(f"\n[2/3] Running nolds parallel poly ({n_workers} workers)...")
    start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        le_nolds_poly = list(executor.map(nolds_lyap_r_poly_worker, args_list))
    results["nolds_poly"] = time.time() - start
    print(f"      Time: {results['nolds_poly']:.3f}s")

    # 3. JAX batch poly (vectorized least squares)
    print("\n[3/4] Running JAX batch poly (vectorized)...")
    # Warmup
    _ = lyap_r_batch(
        solutions, emb_dim=emb_dim, lag=lag, trajectory_len=trajectory_len, fit="poly"
    ).block_until_ready()
    start = time.time()
    le_jax_poly = lyap_r_batch(
        solutions2, emb_dim=emb_dim, lag=lag, trajectory_len=trajectory_len, fit="poly"
    )
    le_jax_poly.block_until_ready()
    results["jax_poly"] = time.time() - start
    print(f"      Time: {results['jax_poly']:.3f}s")

    # 4. JAX batch RANSAC (vectorized robust fitting)
    print("\n[4/4] Running JAX batch RANSAC (vectorized)...")
    key = jax.random.PRNGKey(0)
    # Use None for threshold to auto-compute MAD like sklearn
    # Warmup
    _ = lyap_r_batch(
        solutions,
        emb_dim=emb_dim,
        lag=lag,
        trajectory_len=trajectory_len,
        fit="RANSAC",
        ransac_n_iters=200,
        ransac_threshold=None,
        rng_key=key,
    ).block_until_ready()
    start = time.time()
    le_jax_ransac = lyap_r_batch(
        solutions2,
        emb_dim=emb_dim,
        lag=lag,
        trajectory_len=trajectory_len,
        fit="RANSAC",
        ransac_n_iters=200,
        ransac_threshold=None,
        rng_key=key,
    )
    le_jax_ransac.block_until_ready()
    results["jax_ransac"] = time.time() - start
    print(f"      Time: {results['jax_ransac']:.3f}s")

    # Print results
    baseline = results["nolds_ransac"]
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Implementation':<25} {'Time':>10} {'Per traj':>12} {'Speedup':>10}")
    print("-" * 70)
    print(
        f"{'nolds RANSAC':<25} {results['nolds_ransac']:>9.3f}s "
        f"{results['nolds_ransac'] / n_samples * 1000:>10.1f}ms {'1.0x':>10}"
    )
    print(
        f"{'nolds poly':<25} {results['nolds_poly']:>9.3f}s "
        f"{results['nolds_poly'] / n_samples * 1000:>10.1f}ms "
        f"{baseline / results['nolds_poly']:>9.1f}x"
    )
    print(
        f"{'JAX poly':<25} {results['jax_poly']:>9.3f}s "
        f"{results['jax_poly'] / n_samples * 1000:>10.1f}ms "
        f"{baseline / results['jax_poly']:>9.1f}x"
    )
    print(
        f"{'JAX RANSAC':<25} {results['jax_ransac']:>9.3f}s "
        f"{results['jax_ransac'] / n_samples * 1000:>10.1f}ms "
        f"{baseline / results['jax_ransac']:>9.1f}x"
    )
    print("=" * 70)

    # Verify accuracy
    print("\nAccuracy check (first trajectory):")
    print(f"  nolds RANSAC: {le_nolds_ransac[0]:.6f}  (robust to outliers)")
    print(f"  nolds poly:   {le_nolds_poly[0]:.6f}  (least squares)")
    print(f"  JAX poly:     {float(le_jax_poly[0, 0]):.6f}  (least squares)")
    print(f"  JAX RANSAC:   {float(le_jax_ransac[0, 0]):.6f}  (robust to outliers)")
    print(
        f"\n  JAX poly matches nolds poly: {np.isclose(le_nolds_poly[0], float(le_jax_poly[0, 0]), rtol=1e-3)}"  # type: ignore[arg-type]
    )
    print(
        f"  JAX RANSAC close to nolds RANSAC: {np.isclose(le_nolds_ransac[0], float(le_jax_ransac[0, 0]), rtol=0.5)}"  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    run_benchmark(n_samples=100, n_workers=16)
