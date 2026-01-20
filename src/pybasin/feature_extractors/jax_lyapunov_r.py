# pyright: basic
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax, random, vmap

from pybasin.feature_extractors.jax_feature_utilities import (
    delay_embedding,
    impute,
    rowwise_euclidean,
)


@partial(jax.jit, static_argnums=(1, 2, 3, 5, 6, 7))
def lyap_r_single(
    data: Array,
    emb_dim: int = 10,
    lag: int = 1,
    trajectory_len: int = 20,
    tau: float = 1.0,
    fit: str = "poly",
    ransac_n_iters: int = 100,
    ransac_threshold: float | None = None,
    rng_key: Array | None = None,
) -> Array:
    """Compute largest Lyapunov exponent for a single 1D time series.

    This is a JAX implementation of the Rosenstein algorithm.

    :param data: 1D time series of shape (N,).
    :param emb_dim: Embedding dimension (default: 10).
    :param lag: Lag for delay embedding (default: 1). Must be static for JIT.
    :param trajectory_len: Number of steps to follow divergence (default: 20).
    :param tau: Time step size for normalization (default: 1.0).
    :param fit: Fitting method - "poly" for least squares, "RANSAC" for robust (default: "poly").
    :param ransac_n_iters: Number of RANSAC iterations (default: 100).
    :param ransac_threshold: Inlier threshold for RANSAC. If None, uses MAD (default: None).
    :param rng_key: JAX PRNG key for RANSAC (required if fit="RANSAC").
    :return: Largest Lyapunov exponent (scalar).
    """
    n = data.shape[0]

    min_tsep = compute_min_tsep(data, n)

    orbit = delay_embedding(data, emb_dim, lag)
    m = orbit.shape[0]

    def compute_row_distances(i: Array) -> Array:
        return rowwise_euclidean(orbit, orbit[i])

    dists = vmap(compute_row_distances)(jnp.arange(m))

    mask_range = jnp.arange(m)
    for_masking = jnp.abs(mask_range[:, None] - mask_range[None, :]) <= min_tsep
    dists = jnp.where(for_masking, jnp.inf, dists)

    ntraj = max(m - trajectory_len + 1, 1)

    search_dists = jnp.where(
        (jnp.arange(m)[:, None] < ntraj) & (jnp.arange(m)[None, :] < ntraj),
        dists,
        jnp.inf,
    )

    nb_idx = jnp.argmin(search_dists[:ntraj, :ntraj], axis=1)

    def compute_div_at_k(k: Array) -> Array:
        i_indices = jnp.arange(ntraj) + k
        j_indices = nb_idx + k

        valid_i = i_indices < m
        valid_j = j_indices < m
        valid = valid_i & valid_j

        div_k = dists[i_indices, j_indices]

        div_k_safe = jnp.where((div_k > 0) & valid, div_k, 1.0)
        log_div_k = jnp.log(div_k_safe)
        log_div_k = jnp.where((div_k > 0) & valid, log_div_k, jnp.nan)

        return jnp.nanmean(log_div_k)

    div_traj = vmap(compute_div_at_k)(jnp.arange(trajectory_len))

    ks = jnp.arange(trajectory_len, dtype=jnp.float32)
    finite_mask = jnp.isfinite(div_traj)

    if fit == "RANSAC":
        key = rng_key if rng_key is not None else random.PRNGKey(0)
        slope = _ransac_line_fit(key, ks, div_traj, finite_mask, ransac_n_iters, ransac_threshold)
    else:
        slope = _poly_line_fit(ks, div_traj, finite_mask)

    le = slope / tau

    return le


@partial(jax.jit, static_argnums=(1, 2, 3, 5, 6, 7))
def lyap_r_batch_single_state(
    data: Array,
    emb_dim: int = 10,
    lag: int = 1,
    trajectory_len: int = 20,
    tau: float = 1.0,
    fit: str = "poly",
    ransac_n_iters: int = 100,
    ransac_threshold: float | None = None,
    rng_key: Array | None = None,
) -> Array:
    """Compute Lyapunov exponents for a batch of 1D time series.

    :param data: Batch of time series, shape (B, N) where B is batch size, N is time points.
    :param emb_dim: Embedding dimension.
    :param lag: Lag for delay embedding (must be static for JIT).
    :param trajectory_len: Number of steps to follow divergence.
    :param tau: Time step size for normalization.
    :param fit: Fitting method - "poly" for least squares, "RANSAC" for robust (default: "poly").
    :param ransac_n_iters: Number of RANSAC iterations (default: 100).
    :param ransac_threshold: Inlier threshold for RANSAC. If None, uses MAD (default: None).
    :param rng_key: JAX PRNG key for RANSAC (required if fit="RANSAC").
    :return: Array of Lyapunov exponents, shape (B,).
    """
    batch_size = data.shape[0]

    if fit == "RANSAC":
        key = rng_key if rng_key is not None else random.PRNGKey(0)
        keys = random.split(key, batch_size)

        def compute_single(series: Array, k: Array) -> Array:
            return lyap_r_single(
                series, emb_dim, lag, trajectory_len, tau, fit, ransac_n_iters, ransac_threshold, k
            )

        return vmap(compute_single)(data, keys)

    def compute_single_poly(series: Array) -> Array:
        return lyap_r_single(series, emb_dim, lag, trajectory_len, tau, fit)

    return vmap(compute_single_poly)(data)


@partial(jax.jit, static_argnums=(1, 2, 3, 5, 6, 7))
def lyap_r_batch(
    data: Array,
    emb_dim: int = 10,
    lag: int = 1,
    trajectory_len: int = 20,
    tau: float = 1.0,
    fit: str = "poly",
    ransac_n_iters: int = 100,
    ransac_threshold: float | None = None,
    rng_key: Array | None = None,
) -> Array:
    """Compute Lyapunov exponents for batch of multi-state trajectories.

    This is the main entry point for computing Lyapunov exponents on
    batched trajectory data with multiple state variables.

    :param data: Trajectories of shape (N, B, S) where N is number of time points,
        B is batch size (number of initial conditions), and S is number of state variables.
    :param emb_dim: Embedding dimension.
    :param lag: Lag for delay embedding (must be static for JIT).
    :param trajectory_len: Number of steps to follow divergence.
    :param tau: Time step size for normalization.
    :param fit: Fitting method - "poly" for least squares, "RANSAC" for robust (default: "poly").
    :param ransac_n_iters: Number of RANSAC iterations (default: 100).
    :param ransac_threshold: Inlier threshold for RANSAC. If None, uses MAD (default: None).
    :param rng_key: JAX PRNG key for RANSAC (required if fit="RANSAC").
    :return: Array of Lyapunov exponents, shape (B, S).
    """
    _n_time, batch_size, n_states = data.shape

    data_transposed = jnp.transpose(data, (1, 2, 0))

    if fit == "RANSAC":
        key = rng_key if rng_key is not None else random.PRNGKey(0)
        keys = random.split(key, batch_size * n_states).reshape(batch_size, n_states, 2)

        def compute_for_sample_ransac(sample_data: Array, sample_keys: Array) -> Array:
            def compute_single_state(s: Array, k: Array) -> Array:
                return lyap_r_single(
                    s, emb_dim, lag, trajectory_len, tau, fit, ransac_n_iters, ransac_threshold, k
                )

            return vmap(compute_single_state)(sample_data, sample_keys)

        results = vmap(compute_for_sample_ransac)(data_transposed, keys)
    else:

        def compute_for_sample(sample_data: Array) -> Array:
            return vmap(lambda s: lyap_r_single(s, emb_dim, lag, trajectory_len, tau, fit))(
                sample_data
            )

        results = vmap(compute_for_sample)(data_transposed)

    return results


def lyap_r_batch_with_impute(
    data: Array,
    emb_dim: int = 10,
    lag: int = 1,
    trajectory_len: int = 20,
    tau: float = 1.0,
) -> Array:
    """Compute Lyapunov exponents with automatic imputation of NaN/inf values.

    Same as lyap_r_batch but applies columnwise imputation after computation.

    :param data: Trajectories of shape (N, B, S).
    :param emb_dim: Embedding dimension.
    :param lag: Lag for delay embedding (must be static for JIT).
    :param trajectory_len: Number of steps to follow divergence.
    :param tau: Time step size for normalization.
    :return: Array of Lyapunov exponents, shape (B, S), with NaN/inf imputed.
    """
    features = lyap_r_batch(data, emb_dim, lag, trajectory_len, tau)
    return impute(features)


def _poly_line_fit(ks: Array, div_traj: Array, finite_mask: Array) -> Array:
    """Fit a line using least squares (polynomial fit).

    :param ks: x-values (k indices), shape (N,).
    :param div_traj: y-values (divergence trajectory), shape (N,).
    :param finite_mask: Boolean mask for valid (finite) points, shape (N,).
    :return: Slope of the fitted line.
    """
    n_finite = jnp.sum(finite_mask)

    ks_finite = jnp.where(finite_mask, ks, 0.0)
    div_finite = jnp.where(finite_mask, div_traj, 0.0)

    sum_k = jnp.sum(ks_finite)
    sum_d = jnp.sum(div_finite)
    sum_kk = jnp.sum(ks_finite * ks_finite)
    sum_kd = jnp.sum(ks_finite * div_finite)

    denom = n_finite * sum_kk - sum_k * sum_k
    slope = jnp.where(
        (n_finite >= 2) & (jnp.abs(denom) > 1e-10),
        (n_finite * sum_kd - sum_k * sum_d) / denom,
        jnp.nan,
    )

    return slope


def _fit_line_two_points(x1: Array, y1: Array, x2: Array, y2: Array) -> tuple[Array, Array]:
    """Fit a line y = a*x + b through two points.

    :param x1: First point x-coordinate.
    :param y1: First point y-coordinate.
    :param x2: Second point x-coordinate.
    :param y2: Second point y-coordinate.
    :return: Tuple of (slope a, intercept b).
    """
    a = (y2 - y1) / (x2 - x1 + 1e-10)
    b = y1 - a * x1
    return a, b


def _line_residuals(a: Array, b: Array, x: Array, y: Array) -> Array:
    """Compute absolute residuals for line y = a*x + b.

    :param a: Slope.
    :param b: Intercept.
    :param x: x-values, shape (N,).
    :param y: y-values, shape (N,).
    :return: Absolute residuals, shape (N,).
    """
    y_pred = a * x + b
    return jnp.abs(y - y_pred)


def _ransac_line_fit(
    key: Array,
    ks: Array,
    div_traj: Array,
    finite_mask: Array,
    n_iters: int = 100,
    inlier_threshold: float | Array | None = None,
) -> Array:
    """Fit a line using RANSAC (robust to outliers).

    :param key: JAX PRNG key.
    :param ks: x-values (k indices), shape (N,).
    :param div_traj: y-values (divergence trajectory), shape (N,).
    :param finite_mask: Boolean mask for valid (finite) points, shape (N,).
    :param n_iters: Number of RANSAC iterations.
    :param inlier_threshold: Threshold for considering a point an inlier.
        If None, uses MAD (Median Absolute Deviation) like sklearn.
    :return: Slope of the robustly fitted line.
    """
    n = ks.shape[0]
    n_finite = jnp.sum(finite_mask)

    ks_masked = jnp.where(finite_mask, ks, 0.0)
    div_masked = jnp.where(finite_mask, div_traj, 0.0)

    if inlier_threshold is None:
        div_valid = jnp.where(finite_mask, div_traj, jnp.nan)
        median_y = jnp.nanmedian(div_valid)
        mad = jnp.nanmedian(jnp.abs(div_valid - median_y))
        threshold = mad
    else:
        threshold = inlier_threshold

    def body_fun(i: int, state: tuple) -> tuple:
        key, best_slope, best_intercept, best_inlier_count = state

        key, subkey1, subkey2 = random.split(key, 3)

        idx1 = random.randint(subkey1, (), 0, n)
        idx2 = random.randint(subkey2, (), 0, n)

        x1, y1 = ks_masked[idx1], div_masked[idx1]
        x2, y2 = ks_masked[idx2], div_masked[idx2]

        both_valid = finite_mask[idx1] & finite_mask[idx2] & (idx1 != idx2)

        a, b = _fit_line_two_points(x1, y1, x2, y2)

        residuals = _line_residuals(a, b, ks_masked, div_masked)
        inliers = (residuals < threshold) & finite_mask
        n_inliers = jnp.sum(inliers)

        n_inliers = jnp.where(both_valid, n_inliers, 0)

        is_better = n_inliers > best_inlier_count
        best_slope = jnp.where(is_better, a, best_slope)
        best_intercept = jnp.where(is_better, b, best_intercept)
        best_inlier_count = jnp.where(is_better, n_inliers, best_inlier_count)

        return (key, best_slope, best_intercept, best_inlier_count)

    init_state = (key, jnp.array(0.0), jnp.array(0.0), jnp.array(0))

    _, best_slope, best_intercept, _ = lax.fori_loop(0, n_iters, body_fun, init_state)

    residuals = _line_residuals(best_slope, best_intercept, ks_masked, div_masked)
    inlier_mask = (residuals < threshold) & finite_mask

    final_slope = jnp.where(
        jnp.sum(inlier_mask) >= 2,
        _poly_line_fit(ks, div_traj, inlier_mask),  # type: ignore[arg-type]
        best_slope,
    )

    final_slope = jnp.where(n_finite >= 2, final_slope, jnp.nan)  # type: ignore[arg-type]

    return final_slope


def compute_min_tsep(data: Array, n: int) -> Array:
    """Compute minimum temporal separation using mean frequency.

    :param data: 1D time series.
    :param n: Length of data.
    :return: Minimum temporal separation (as JAX Array).
    """
    max_tsep_factor = 0.25

    f = jnp.fft.rfft(data, n * 2 - 1)
    freqs = jnp.fft.rfftfreq(n * 2 - 1)
    psd = jnp.abs(f) ** 2

    mf = jnp.sum(freqs[1:] * psd[1:]) / (jnp.sum(psd[1:]) + 1e-10)
    min_tsep = jnp.ceil(1.0 / (mf + 1e-10)).astype(jnp.int32)

    max_tsep = jnp.array(max_tsep_factor * n, dtype=jnp.int32)
    min_tsep = jnp.minimum(min_tsep, max_tsep)
    min_tsep = jnp.maximum(min_tsep, 1)

    return min_tsep
