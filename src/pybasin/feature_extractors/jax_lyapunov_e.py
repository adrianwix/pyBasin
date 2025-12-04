# pyright: basic
"""JAX-based Lyapunov exponent and correlation dimension calculators with GPU acceleration.

This module provides vectorized implementations of:
1. Rosenstein algorithm (lyap_r) - estimates the largest Lyapunov exponent
2. Eckmann algorithm (lyap_e) - estimates multiple Lyapunov exponents
3. Grassberger-Procaccia algorithm (corr_dim) - estimates correlation dimension

All are optimized for batch processing of many trajectories simultaneously on GPU.

References:
    M. T. Rosenstein, J. J. Collins, and C. J. De Luca,
    "A practical method for calculating largest Lyapunov exponents from
    small data sets," Physica D: Nonlinear Phenomena, vol. 65, no. 1,
    pp. 117-134, 1993.

    J. P. Eckmann, S. O. Kamphorst, D. Ruelle, and S. Ciliberto,
    "Liapunov exponents from time series," Physical Review A,
    vol. 34, no. 6, pp. 4971-4979, 1986.

    P. Grassberger and I. Procaccia, "Characterization of strange
    attractors," Physical review letters, vol. 50, no. 5, p. 346, 1983.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import cpu_count

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from pybasin.feature_extractors.jax_feature_utilities import delay_embedding

# =============================================================================
# Eckmann Algorithm (lyap_e) - Multiple Lyapunov Exponents
# =============================================================================


def rowwise_chebyshev(x: Array, y: Array) -> Array:
    """Compute Chebyshev (L-infinity) distance from each row of x to vector y.

    Args:
        x: Matrix of shape (M, D)
        y: Vector of shape (D,)

    Returns:
        Distances of shape (M,)
    """
    return jnp.max(jnp.abs(x - y), axis=1)


def lyap_e_single(
    data: np.ndarray,
    emb_dim: int = 10,
    matrix_dim: int = 4,
    min_nb: int = 8,
    min_tsep: int = 0,
    tau: float = 1.0,
) -> np.ndarray:
    """Compute multiple Lyapunov exponents for a single 1D time series.

    This is a NumPy implementation of the Eckmann algorithm, which is faster
    than JAX for this inherently sequential algorithm due to NumPy's optimized
    LAPACK calls for small matrices.

    Args:
        data: 1D time series of shape (N,)
        emb_dim: Embedding dimension (default: 10). Must satisfy (emb_dim-1) % (matrix_dim-1) == 0
        matrix_dim: Matrix dimension for Jacobian estimation (default: 4)
        min_nb: Minimal number of neighbors (default: 8)
        min_tsep: Minimal temporal separation between neighbors (default: 0)
        tau: Time step size for normalization (default: 1.0)

    Returns:
        Array of matrix_dim Lyapunov exponents
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    m = (emb_dim - 1) // (matrix_dim - 1)

    # Build orbit (delay embedding)
    orbit = np.array([data[i : i + emb_dim] for i in range(n - emb_dim - m + 1)])
    n_orbit = len(orbit)

    old_q = np.identity(matrix_dim)
    lexp = np.zeros(matrix_dim, dtype=np.float64)
    lexp_counts = np.zeros(matrix_dim, dtype=np.float64)

    for i in range(n_orbit):
        # Chebyshev distance
        diffs = np.max(np.abs(orbit - orbit[i]), axis=1)
        diffs[i] = np.inf

        # Temporal masking
        mask_from = max(0, i - min_tsep)
        mask_to = min(len(diffs), i + min_tsep + 1)
        diffs[mask_from:mask_to] = np.inf

        # Find neighbors
        indices = np.argsort(diffs)
        idx = indices[min_nb - 1]
        r = diffs[idx]

        if np.isinf(r):
            continue

        indices = np.where(diffs <= r)[0]

        # Build least squares matrices
        mat_x = np.array([data[j : j + emb_dim : m] for j in indices])
        mat_x -= data[i : i + emb_dim : m]
        vec_beta = data[indices + matrix_dim * m] - data[i + matrix_dim * m]

        # Solve least squares
        a, _, _, _ = np.linalg.lstsq(mat_x, vec_beta, rcond=-1)

        # Build T matrix
        mat_t = np.zeros((matrix_dim, matrix_dim))
        mat_t[:-1, 1:] = np.identity(matrix_dim - 1)
        mat_t[-1] = a

        # QR decomposition
        mat_q, mat_r = np.linalg.qr(mat_t @ old_q)

        # Force positive diagonal
        sign_diag = np.sign(np.diag(mat_r))
        sign_diag[sign_diag == 0] = 1
        mat_q = mat_q @ np.diag(sign_diag)
        mat_r = np.diag(sign_diag) @ mat_r

        old_q = mat_q

        # Accumulate Lyapunov exponents
        diag_r = np.diag(mat_r)
        idx_pos = np.where(diag_r > 0)[0]
        lexp[idx_pos] += np.log(diag_r[idx_pos])
        lexp_counts[idx_pos] += 1

    # Normalize
    idx = np.where(lexp_counts > 0)[0]
    lexp[idx] /= lexp_counts[idx]
    lexp[lexp_counts == 0] = np.inf
    lexp /= tau
    lexp /= m

    return lexp


def _lyap_e_worker(args: tuple) -> np.ndarray:
    """Worker function for parallel lyap_e computation."""
    data, emb_dim, matrix_dim, min_nb, min_tsep, tau = args
    return lyap_e_single(data, emb_dim, matrix_dim, min_nb, min_tsep, tau)


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def lyap_e_single_jax(
    data: Array,
    emb_dim: int = 10,
    matrix_dim: int = 4,
    min_nb: int = 8,
    min_tsep: int = 0,
    tau: float = 1.0,
) -> Array:
    """Compute multiple Lyapunov exponents for a single 1D time series using JAX.

    NOTE: This JAX implementation is slower than the numpy version due to
    JAX's fori_loop overhead on sequential operations. Use lyap_e_single
    (numpy-based) for better performance.

    This is a JAX implementation of the Eckmann algorithm using fori_loop
    for better performance on sequential operations.

    Args:
        data: 1D time series of shape (N,)
        emb_dim: Embedding dimension (default: 10). Must satisfy (emb_dim-1) % (matrix_dim-1) == 0
        matrix_dim: Matrix dimension for Jacobian estimation (default: 4)
        min_nb: Minimal number of neighbors (default: 8)
        min_tsep: Minimal temporal separation between neighbors (default: 0)
        tau: Time step size for normalization (default: 1.0)

    Returns:
        Array of matrix_dim Lyapunov exponents
    """
    m = (emb_dim - 1) // (matrix_dim - 1)

    # Build orbit matrix (delay embedding with lag=1)
    orbit = delay_embedding(data[:-m], emb_dim, lag=1)
    n_orbit = orbit.shape[0]

    # Pre-compute stepped data for efficient access
    step_offsets = jnp.arange(matrix_dim) * m
    stepped_data = data[jnp.arange(n_orbit)[:, None] + step_offsets[None, :]]
    beta_data = data[jnp.arange(n_orbit) + matrix_dim * m]

    # Create index array for temporal masking
    all_indices = jnp.arange(n_orbit)

    def body_fn(i, carry):
        """Single step of the QR iteration."""
        old_q, lexp, lexp_counts = carry

        # Compute Chebyshev distances from orbit[i] to all orbit vectors
        diffs = jnp.max(jnp.abs(orbit - orbit[i]), axis=1)

        # Mask self and temporally close neighbors
        time_mask = jnp.abs(all_indices - i) <= min_tsep
        diffs = jnp.where(time_mask, jnp.inf, diffs)

        # Find the min_nb-th nearest neighbor distance (radius)
        sorted_diffs = jnp.sort(diffs)
        r = sorted_diffs[min_nb - 1]

        # Get neighbor mask (all points within radius r)
        neighbor_mask = diffs <= r

        # Build X matrix and beta vector for least squares
        orbit_i_stepped = stepped_data[i]
        beta_i = beta_data[i]

        x_diff = stepped_data - orbit_i_stepped
        beta_diff = beta_data - beta_i

        # Apply neighbor mask
        x_masked = jnp.where(neighbor_mask[:, None], x_diff, 0.0)
        beta_masked = jnp.where(neighbor_mask, beta_diff, 0.0)

        # Solve least squares using normal equations
        xtx = x_masked.T @ x_masked + 1e-10 * jnp.eye(matrix_dim)
        xty = x_masked.T @ beta_masked
        a = jnp.linalg.solve(xtx, xty)

        # Build T matrix
        t_mat = jnp.zeros((matrix_dim, matrix_dim))
        t_mat = t_mat.at[:-1, 1:].set(jnp.eye(matrix_dim - 1))
        t_mat = t_mat.at[-1, :].set(a)

        # QR decomposition
        mat_q, mat_r = jnp.linalg.qr(t_mat @ old_q)

        # Force positive diagonal
        sign_diag = jnp.sign(jnp.diag(mat_r))
        sign_diag = jnp.where(sign_diag == 0, 1.0, sign_diag)
        mat_q = mat_q @ jnp.diag(sign_diag)
        mat_r = jnp.diag(sign_diag) @ mat_r

        # Extract log of diagonal
        diag_r = jnp.diag(mat_r)
        log_diag = jnp.where(diag_r > 0, jnp.log(diag_r), 0.0)
        valid_diag = diag_r > 0

        # Update accumulators
        new_lexp = lexp + jnp.where(valid_diag, log_diag, 0.0)
        new_counts = lexp_counts + valid_diag.astype(jnp.float32)

        # Skip if radius is infinite (not enough neighbors)
        skip = jnp.isinf(r)
        new_q = jnp.where(skip, old_q, mat_q)
        new_lexp = jnp.where(skip, lexp, new_lexp)
        new_counts = jnp.where(skip, lexp_counts, new_counts)

        return (new_q, new_lexp, new_counts)

    # Initialize and run loop
    init_carry = (jnp.eye(matrix_dim), jnp.zeros(matrix_dim), jnp.zeros(matrix_dim))
    _, final_lexp, final_counts = jax.lax.fori_loop(0, n_orbit, body_fn, init_carry)

    # Normalize exponents
    normalized = jnp.where(final_counts > 0, final_lexp / final_counts, jnp.inf)
    normalized = normalized / tau / m

    return normalized


def lyap_e_batch_single_state(
    data: Array | np.ndarray,
    emb_dim: int = 10,
    matrix_dim: int = 4,
    min_nb: int = 8,
    min_tsep: int = 0,
    tau: float = 1.0,
    n_workers: int | None = None,
) -> Array:
    """Compute multiple Lyapunov exponents for a batch of 1D time series.

    Uses parallel numpy implementation with ProcessPoolExecutor for speed.

    Args:
        data: Batch of time series, shape (B, N) where B is batch size, N is time points
        emb_dim: Embedding dimension
        matrix_dim: Matrix dimension
        min_nb: Minimal number of neighbors
        min_tsep: Minimal temporal separation
        tau: Time step size for normalization
        n_workers: Number of parallel workers (default: min(cpu_count, 16))

    Returns:
        Array of Lyapunov exponents, shape (B, matrix_dim)
    """
    data_np = np.asarray(data)
    batch_size = data_np.shape[0]

    if n_workers is None:
        n_workers = min(cpu_count(), 16)

    args_list = [
        (data_np[i], emb_dim, matrix_dim, min_nb, min_tsep, tau) for i in range(batch_size)
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_lyap_e_worker, args_list))

    return jnp.array(results)


def lyap_e_batch(
    data: Array | np.ndarray,
    emb_dim: int = 10,
    matrix_dim: int = 4,
    min_nb: int = 8,
    min_tsep: int = 0,
    tau: float = 1.0,
    n_workers: int | None = None,
) -> Array:
    """Compute multiple Lyapunov exponents for batch of multi-state trajectories.

    Uses parallel numpy implementation with ProcessPoolExecutor for speed.
    This is ~20x faster than the JAX implementation due to NumPy's optimized
    LAPACK calls for small matrices and multiprocessing parallelism.

    Args:
        data: Trajectories of shape (N, B, S) where:
            - N: number of time points
            - B: batch size (number of initial conditions)
            - S: number of state variables
        emb_dim: Embedding dimension
        matrix_dim: Matrix dimension
        min_nb: Minimal number of neighbors
        min_tsep: Minimal temporal separation
        tau: Time step size for normalization
        n_workers: Number of parallel workers (default: min(cpu_count, 16))

    Returns:
        Array of Lyapunov exponents, shape (B, S, matrix_dim)
    """
    data_np = np.asarray(data)
    n_time, batch_size, n_states = data_np.shape

    if n_workers is None:
        n_workers = min(cpu_count(), 16)

    # Prepare all tasks
    args_list = []
    for b in range(batch_size):
        for s in range(n_states):
            args_list.append((data_np[:, b, s], emb_dim, matrix_dim, min_nb, min_tsep, tau))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_lyap_e_worker, args_list))

    # Reshape results to (B, S, matrix_dim)
    results_array = np.array(results).reshape(batch_size, n_states, matrix_dim)
    return jnp.array(results_array)


def lyap_e_batch_with_impute(
    data: Array,
    emb_dim: int = 10,
    matrix_dim: int = 4,
    min_nb: int = 8,
    min_tsep: int = 0,
    tau: float = 1.0,
) -> Array:
    """Compute multiple Lyapunov exponents with automatic imputation.

    Same as lyap_e_batch but applies imputation after computation.

    Args:
        data: Trajectories of shape (N, B, S)
        emb_dim: Embedding dimension
        matrix_dim: Matrix dimension
        min_nb: Minimal number of neighbors
        min_tsep: Minimal temporal separation
        tau: Time step size for normalization

    Returns:
        Array of Lyapunov exponents, shape (B, S, matrix_dim), with NaN/inf imputed
    """
    from pybasin.feature_extractors.jax_feature_utilities import impute

    features = lyap_e_batch(data, emb_dim, matrix_dim, min_nb, min_tsep, tau)
    # Reshape to 2D for imputation, then reshape back
    original_shape = features.shape
    features_2d = features.reshape(-1, matrix_dim)
    imputed = impute(features_2d)
    return imputed.reshape(original_shape)
