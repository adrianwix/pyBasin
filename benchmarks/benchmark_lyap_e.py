# pyright: basic
"""Benchmark lyap_e implementations: nolds, JAX, and parallel numpy."""

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
import multiprocessing
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Suppress pkg_resources deprecation warning from nolds
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import jax
import jax.numpy as jnp
import nolds
import numpy as np

from pybasin.feature_extractors.jax_lyapunov_e import lyap_e_batch, lyap_e_single_jax

# Suppress other warnings
warnings.filterwarnings("ignore")
logging.getLogger("jax").setLevel(logging.CRITICAL)


def _make_pmap_kernel(devices):
    """Create a pmap kernel for the given devices."""

    @partial(jax.pmap, devices=devices, static_broadcasted_argnums=(1, 2, 3, 4, 5))
    def lyap_e_pmap_kernel(data, emb_dim, matrix_dim, min_nb, min_tsep, tau):
        """JAX pmap kernel for multi-core execution."""

        def compute_for_sample(sample):
            return jax.vmap(
                lambda s: lyap_e_single_jax(s, emb_dim, matrix_dim, min_nb, min_tsep, tau)
            )(sample)

        return jax.vmap(compute_for_sample)(data)

    return lyap_e_pmap_kernel


# JAX vmap version (single-threaded)
@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def lyap_e_batch_jax_vmap(data, emb_dim=10, matrix_dim=4, min_nb=8, min_tsep=0, tau=1.0):
    """JAX vmap implementation (single-threaded)."""
    data_t = jnp.transpose(data, (1, 2, 0))

    def compute_for_sample(sample):
        return jax.vmap(lambda s: lyap_e_single_jax(s, emb_dim, matrix_dim, min_nb, min_tsep, tau))(
            sample
        )

    return jax.vmap(compute_for_sample)(data_t)


def lyap_e_batch_jax_pmap(
    data, emb_dim=10, matrix_dim=4, min_nb=8, min_tsep=0, tau=1.0, devices=None
):
    """JAX pmap implementation (multi-core CPU).

    Args:
        data: Trajectories of shape (N, B, S)
        emb_dim: Embedding dimension
        matrix_dim: Matrix dimension
        min_nb: Minimal number of neighbors
        min_tsep: Minimal temporal separation
        tau: Time step size
        devices: List of JAX devices to use. If None, uses all available devices.
    """
    if devices is None:
        devices = jax.devices("cpu")

    n_devices = len(devices)
    data_t = jnp.transpose(data, (1, 2, 0))  # (B, S, N)
    batch_size = data_t.shape[0]

    # Pad to be divisible by n_devices
    pad_size = (n_devices - batch_size % n_devices) % n_devices
    if pad_size > 0:
        data_t = jnp.concatenate(
            [data_t, jnp.zeros((pad_size, data_t.shape[1], data_t.shape[2]))], axis=0
        )

    # Reshape for pmap: (devices, batch_per_device, S, N)
    batch_per_device = data_t.shape[0] // n_devices
    data_pmap = data_t.reshape(n_devices, batch_per_device, data_t.shape[1], data_t.shape[2])

    # Create and run pmap kernel with specified devices
    pmap_kernel = _make_pmap_kernel(devices)
    result = pmap_kernel(data_pmap, emb_dim, matrix_dim, min_nb, min_tsep, tau)

    # Reshape back and remove padding
    result = result.reshape(-1, result.shape[2], result.shape[3])
    if pad_size > 0:
        result = result[:batch_size]

    return result


def nolds_worker(args):
    """Worker function for parallel nolds."""
    data, emb_dim, matrix_dim, min_nb = args
    return nolds.lyap_e(data, emb_dim=emb_dim, matrix_dim=matrix_dim, min_nb=min_nb)


def jax_worker_init():
    """Initialize JAX to use CPU only in worker processes."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def jax_worker(args):
    """Worker function for parallel JAX (each process runs JAX on CPU)."""
    data, emb_dim, matrix_dim, min_nb, min_tsep, tau = args
    result = lyap_e_single_jax(jnp.array(data), emb_dim, matrix_dim, min_nb, min_tsep, tau)
    return np.array(result)


def lyap_e_batch_jax_multiprocess(
    data, emb_dim=10, matrix_dim=4, min_nb=8, min_tsep=0, tau=1.0, n_workers=None
):
    """JAX with ProcessPoolExecutor (each worker runs JAX on CPU)."""
    if n_workers is None:
        n_workers = cpu_count()

    data_np = np.array(data)
    n_time, n_samples, n_states = data_np.shape

    args_list = [
        (data_np[:, i, 0], emb_dim, matrix_dim, min_nb, min_tsep, tau) for i in range(n_samples)
    ]

    # Use spawn context to avoid inheriting parent's CUDA context
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx, initializer=jax_worker_init
    ) as executor:
        results = list(executor.map(jax_worker, args_list))

    return np.array(results)[:, None, :]


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


def run_benchmark(n_samples=100, n_workers=16, n_jax_devices=None):
    """Run the full benchmark suite.

    Args:
        n_samples: Number of trajectories to benchmark
        n_workers: Number of ProcessPool workers for parallel implementations
        n_jax_devices: Number of JAX devices for pmap. If None, uses all available CPU devices.
    """
    all_cpu_devices = jax.devices("cpu")
    devices = all_cpu_devices if n_jax_devices is None else all_cpu_devices[:n_jax_devices]

    print(f"JAX CPU devices available: {len(all_cpu_devices)}")
    print(f"JAX CPU devices for pmap: {len(devices)}")
    print(f"ProcessPool workers: {n_workers}")
    print(f"System CPU count: {cpu_count()}")

    print("=" * 70)
    print(f"BENCHMARK: lyap_e implementations ({n_samples} trajectories)")
    print("=" * 70)

    # Generate data
    solutions = generate_logistic_map_batch(n_samples=n_samples, n_time=500)
    solutions2 = solutions + 0.001  # Slightly different for timing
    solutions_np = np.array(solutions2)
    print(f"Solutions shape: {solutions.shape}")

    emb_dim, matrix_dim = 10, 4
    min_nb = min(2 * matrix_dim, matrix_dim + 4)

    results = {}

    # 1. nolds sequential
    print("\n[1/5] Running nolds (sequential)...")
    start = time.time()
    le_nolds_seq = [
        nolds.lyap_e(solutions_np[:, i, 0], emb_dim=emb_dim, matrix_dim=matrix_dim)
        for i in range(n_samples)
    ]
    results["nolds_seq"] = time.time() - start
    print(f"      Time: {results['nolds_seq']:.3f}s")

    # 2. nolds parallel
    print(f"\n[2/5] Running nolds (parallel, {n_workers} workers)...")
    args_list = [(solutions_np[:, i, 0], emb_dim, matrix_dim, min_nb) for i in range(n_samples)]
    start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        list(executor.map(nolds_worker, args_list))
    results["nolds_par"] = time.time() - start
    print(f"      Time: {results['nolds_par']:.3f}s")

    # 3. JAX vmap (single-threaded)
    print("\n[3/5] Running JAX vmap (single-threaded)...")
    # Warmup
    _ = lyap_e_batch_jax_vmap(
        solutions, emb_dim=emb_dim, matrix_dim=matrix_dim, min_nb=min_nb
    ).block_until_ready()
    start = time.time()
    le_jax_vmap = lyap_e_batch_jax_vmap(
        solutions2, emb_dim=emb_dim, matrix_dim=matrix_dim, min_nb=min_nb
    )
    le_jax_vmap.block_until_ready()
    results["jax_vmap"] = time.time() - start
    print(f"      Time: {results['jax_vmap']:.3f}s")

    # 4. JAX pmap (multi-core)
    print(f"\n[4/6] Running JAX pmap ({len(devices)} devices)...")
    # Warmup
    _ = lyap_e_batch_jax_pmap(
        solutions, emb_dim=emb_dim, matrix_dim=matrix_dim, min_nb=min_nb, devices=devices
    ).block_until_ready()
    start = time.time()
    le_jax_pmap = lyap_e_batch_jax_pmap(
        solutions2, emb_dim=emb_dim, matrix_dim=matrix_dim, min_nb=min_nb, devices=devices
    )
    le_jax_pmap.block_until_ready()
    results["jax_pmap"] = time.time() - start
    print(f"      Time: {results['jax_pmap']:.3f}s")

    # 5. JAX multiprocess (ProcessPoolExecutor with JAX workers)
    print(f"\n[5/6] Running JAX multiprocess ({n_workers} workers)...")
    start = time.time()
    le_jax_mp = lyap_e_batch_jax_multiprocess(
        solutions2, emb_dim=emb_dim, matrix_dim=matrix_dim, min_nb=min_nb, n_workers=n_workers
    )
    results["jax_multiprocess"] = time.time() - start
    print(f"      Time: {results['jax_multiprocess']:.3f}s")

    # 6. Parallel numpy (our implementation)
    print(f"\n[6/6] Running parallel numpy ({n_workers} workers)...")
    start = time.time()
    le_np = lyap_e_batch(solutions2, emb_dim=emb_dim, matrix_dim=matrix_dim, min_nb=min_nb)
    results["numpy_par"] = time.time() - start
    print(f"      Time: {results['numpy_par']:.3f}s")

    # Print results
    baseline = results["nolds_seq"]
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Implementation':<25} {'Time':>10} {'Per traj':>12} {'Speedup':>10}")
    print("-" * 70)
    print(
        f"{'nolds sequential':<25} {results['nolds_seq']:>9.3f}s "
        f"{results['nolds_seq'] / n_samples * 1000:>10.1f}ms {'1.0x':>10}"
    )
    print(
        f"{'nolds parallel':<25} {results['nolds_par']:>9.3f}s "
        f"{results['nolds_par'] / n_samples * 1000:>10.1f}ms "
        f"{baseline / results['nolds_par']:>9.1f}x"
    )
    print(
        f"{'JAX vmap (1 core)':<25} {results['jax_vmap']:>9.3f}s "
        f"{results['jax_vmap'] / n_samples * 1000:>10.1f}ms "
        f"{baseline / results['jax_vmap']:>9.1f}x"
    )
    print(
        f"{'JAX pmap (multi-core)':<25} {results['jax_pmap']:>9.3f}s "
        f"{results['jax_pmap'] / n_samples * 1000:>10.1f}ms "
        f"{baseline / results['jax_pmap']:>9.1f}x"
    )
    print(
        f"{'JAX multiprocess':<25} {results['jax_multiprocess']:>9.3f}s "
        f"{results['jax_multiprocess'] / n_samples * 1000:>10.1f}ms "
        f"{baseline / results['jax_multiprocess']:>9.1f}x"
    )
    print(
        f"{'Parallel numpy':<25} {results['numpy_par']:>9.3f}s "
        f"{results['numpy_par'] / n_samples * 1000:>10.1f}ms "
        f"{baseline / results['numpy_par']:>9.1f}x"
    )
    print("=" * 70)

    # Verify accuracy
    print("\nAccuracy check (first trajectory):")
    print(f"  nolds:          {le_nolds_seq[0]}")
    print(f"  JAX vmap:       {np.array(le_jax_vmap[0, 0])}")
    print(f"  JAX pmap:       {np.array(le_jax_pmap[0, 0])}")
    print(f"  JAX multiproc:  {le_jax_mp[0, 0]}")
    print(f"  numpy par:      {np.array(le_np[0, 0])}")

    return results


if __name__ == "__main__":
    run_benchmark(n_samples=100, n_workers=16)
