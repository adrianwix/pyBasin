# pyright: basic
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, vmap

from pybasin.feature_extractors.jax_feature_utilities import delay_embedding, rowwise_euclidean

# =============================================================================
# Grassberger-Procaccia Algorithm (corr_dim) - Correlation Dimension
# =============================================================================


def logarithmic_r_numpy(min_r: float, max_r: float, factor: float) -> np.ndarray:
    """Create logarithmically spaced radius values (NumPy version for static computation).

    Args:
        min_r: Minimum radius value
        max_r: Maximum radius value
        factor: Multiplication factor (must be > 1)

    Returns:
        NumPy array of logarithmically spaced values from min_r to max_r
    """
    import numpy as np

    max_i = int(np.floor(np.log(max_r / min_r) / np.log(factor)))
    return np.array([min_r * (factor**i) for i in range(max_i + 1)])


def _corr_dim_core(
    data: Array,
    orbit: Array,
    rvals: Array,
) -> Array:
    """Core correlation dimension computation (JIT-compilable).

    Args:
        data: Original 1D time series (unused, kept for signature consistency)
        orbit: Delay-embedded orbit matrix of shape (M, emb_dim)
        rvals: Array of radius values to use

    Returns:
        Correlation dimension (scalar)
    """
    n = orbit.shape[0]

    # Compute pairwise Euclidean distances
    def compute_row_distances(i: Array) -> Array:
        return rowwise_euclidean(orbit, orbit[i])

    dists = vmap(compute_row_distances)(jnp.arange(n))

    # Zero out diagonal (self-distances) to exclude self-matches
    dists = dists.at[jnp.diag_indices(n)].set(jnp.inf)

    # Compute correlation sums for each radius
    def compute_csum(r: Array) -> Array:
        count = jnp.sum(dists <= r)
        return count / (n * (n - 1))

    csums = vmap(compute_csum)(rvals)

    # Filter out zero csums and fit line in log-log space
    nonzero_mask = csums > 0
    log_r = jnp.log(rvals)
    log_c = jnp.where(nonzero_mask, jnp.log(csums + 1e-30), jnp.nan)

    # Linear regression: log(C) = D * log(r) + intercept
    n_valid = jnp.sum(nonzero_mask)

    log_r_valid = jnp.where(nonzero_mask, log_r, 0.0)
    log_c_valid = jnp.where(nonzero_mask, log_c, 0.0)

    sum_r = jnp.sum(log_r_valid)
    sum_c = jnp.sum(log_c_valid)
    sum_rr = jnp.sum(log_r_valid * log_r_valid)
    sum_rc = jnp.sum(log_r_valid * log_c_valid)

    denom = n_valid * sum_rr - sum_r * sum_r
    slope = jnp.where(
        (n_valid >= 2) & (jnp.abs(denom) > 1e-10),
        (n_valid * sum_rc - sum_r * sum_c) / denom,
        jnp.nan,
    )

    return slope


@partial(jax.jit, static_argnums=(1, 2, 3))
def corr_dim_single(
    data: Array,
    emb_dim: int = 4,
    lag: int = 1,
    n_rvals: int = 50,
) -> Array:
    """Compute correlation dimension for a single 1D time series.

    This is a JAX implementation of the Grassberger-Procaccia algorithm.

    The correlation dimension is a characteristic measure that can be used
    to describe the geometry of chaotic attractors. It is defined using the
    correlation sum C(r) which is the fraction of pairs of points in the
    phase space whose distance is smaller than r.

    If C(r) ~ r^D, then D is the correlation dimension.

    Args:
        data: 1D time series of shape (N,)
        emb_dim: Embedding dimension (default: 4)
        lag: Lag for delay embedding (default: 1)
        n_rvals: Number of radius values to use (default: 50)

    Returns:
        Correlation dimension (scalar)
    """
    # Create delay embedding
    orbit = delay_embedding(data, emb_dim, lag)

    # Compute rvals based on data statistics (logarithmically spaced)
    sd = jnp.std(data, ddof=1)
    min_r = 0.1 * sd
    max_r = 0.5 * sd
    # Use linspace in log space to get logarithmically spaced values
    log_min = jnp.log(min_r + 1e-10)
    log_max = jnp.log(max_r + 1e-10)
    rvals = jnp.exp(jnp.linspace(log_min, log_max, n_rvals))

    return _corr_dim_core(data, orbit, rvals)


@partial(jax.jit, static_argnums=(1, 2, 3))
def corr_dim_batch_single_state(
    data: Array,
    emb_dim: int = 4,
    lag: int = 1,
    n_rvals: int = 50,
) -> Array:
    """Compute correlation dimension for a batch of 1D time series.

    Args:
        data: Batch of time series, shape (B, N) where B is batch size, N is time points
        emb_dim: Embedding dimension
        lag: Lag for delay embedding
        n_rvals: Number of radius values to use

    Returns:
        Array of correlation dimensions, shape (B,)
    """

    def compute_single(series: Array) -> Array:
        return corr_dim_single(series, emb_dim, lag, n_rvals)

    return vmap(compute_single)(data)


@partial(jax.jit, static_argnums=(1, 2, 3))
def corr_dim_batch(
    data: Array,
    emb_dim: int = 4,
    lag: int = 1,
    n_rvals: int = 50,
) -> Array:
    """Compute correlation dimension for batch of multi-state trajectories.

    This is the main entry point for computing correlation dimension on
    batched trajectory data with multiple state variables.

    Args:
        data: Trajectories of shape (N, B, S) where:
            - N: number of time points
            - B: batch size (number of initial conditions)
            - S: number of state variables
        emb_dim: Embedding dimension
        lag: Lag for delay embedding
        n_rvals: Number of radius values to use

    Returns:
        Array of correlation dimensions, shape (B, S)
    """
    _n_time, _batch_size, _n_states = data.shape

    # Transpose to (B, S, N)
    data_transposed = jnp.transpose(data, (1, 2, 0))

    def compute_for_sample(sample_data: Array) -> Array:
        # sample_data has shape (S, N)
        return vmap(lambda s: corr_dim_single(s, emb_dim, lag, n_rvals))(sample_data)

    # Result has shape (B, S)
    results = vmap(compute_for_sample)(data_transposed)

    return results


def corr_dim_batch_with_impute(
    data: Array,
    emb_dim: int = 4,
    lag: int = 1,
    n_rvals: int = 50,
) -> Array:
    """Compute correlation dimension with automatic imputation.

    Same as corr_dim_batch but applies imputation after computation.

    Args:
        data: Trajectories of shape (N, B, S)
        emb_dim: Embedding dimension
        lag: Lag for delay embedding
        n_rvals: Number of radius values to use

    Returns:
        Array of correlation dimensions, shape (B, S), with NaN/inf imputed
    """
    from pybasin.feature_extractors.jax_feature_utilities import impute

    features = corr_dim_batch(data, emb_dim, lag, n_rvals)
    return impute(features)
