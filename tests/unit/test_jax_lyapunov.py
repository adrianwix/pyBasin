import jax.numpy as jnp
import numpy as np
import pytest

from pybasin.feature_extractors.jax_corr_dim import corr_dim_batch, corr_dim_single
from pybasin.feature_extractors.jax_feature_utilities import delay_embedding
from pybasin.feature_extractors.jax_lyapunov_e import lyap_e_batch, lyap_e_single
from pybasin.feature_extractors.jax_lyapunov_r import lyap_r_batch, lyap_r_single


class TestDelayEmbedding:
    def test_basic_embedding(self):
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        emb_dim = 3
        lag = 1

        embedded = delay_embedding(data, emb_dim, lag)

        assert embedded.shape == (8, 3)
        np.testing.assert_array_almost_equal(embedded[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(embedded[1], [2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(embedded[-1], [8.0, 9.0, 10.0])

    def test_embedding_with_larger_lag(self):
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        emb_dim = 3
        lag = 2

        embedded = delay_embedding(data, emb_dim, lag)

        assert embedded.shape == (6, 3)
        np.testing.assert_array_almost_equal(embedded[0], [1.0, 3.0, 5.0])
        np.testing.assert_array_almost_equal(embedded[1], [2.0, 4.0, 6.0])

    def test_embedding_shape_formula(self):
        n = 100
        data = jnp.arange(n, dtype=jnp.float32)

        for emb_dim in [2, 3, 5]:
            for lag in [1, 2, 3]:
                embedded = delay_embedding(data, emb_dim, lag)
                expected_len = n - (emb_dim - 1) * lag
                assert embedded.shape == (expected_len, emb_dim)


class TestLyapRSingle:
    @pytest.fixture
    def logistic_map_data(self):
        def logistic_map(r: float, x0: float, n: int) -> jnp.ndarray:
            x = jnp.zeros(n)
            x = x.at[0].set(x0)
            for i in range(1, n):
                x = x.at[i].set(r * x[i - 1] * (1 - x[i - 1]))
            return x

        return logistic_map(3.9, 0.1, 2000)

    def test_logistic_map_positive_exponent(self, logistic_map_data):
        result = lyap_r_single(logistic_map_data, emb_dim=10, lag=2, trajectory_len=20, tau=1.0)

        assert jnp.isfinite(result)
        assert result > 0

    def test_simple_periodic_near_zero_exponent(self):
        t = jnp.linspace(0, 100, 5000)
        data = jnp.sin(t)

        result = lyap_r_single(data, emb_dim=5, lag=10, trajectory_len=50, tau=0.02)

        assert jnp.isfinite(result)
        assert jnp.abs(result) < 0.5

    def test_output_is_scalar(self, logistic_map_data):
        result = lyap_r_single(logistic_map_data, emb_dim=10, lag=2, trajectory_len=20, tau=1.0)
        assert result.shape == ()


class TestLyapRBatch:
    @pytest.fixture
    def batch_logistic_data(self):
        def logistic_map(r: float, x0: float, n: int) -> np.ndarray:
            x = np.zeros(n)
            x[0] = x0
            for i in range(1, n):
                x[i] = r * x[i - 1] * (1 - x[i - 1])
            return x

        batch_size = 4
        n_samples = 3
        n_points = 1000
        data = np.zeros((n_points, batch_size, n_samples))
        for b in range(batch_size):
            for s in range(n_samples):
                r = 3.8 + 0.05 * b
                x0 = 0.1 + 0.01 * s
                data[:, b, s] = logistic_map(r, x0, n_points)

        return jnp.array(data)

    def test_batch_output_shape(self, batch_logistic_data):
        result = lyap_r_batch(batch_logistic_data, emb_dim=10, lag=2, trajectory_len=20, tau=1.0)

        n_points, batch_size, n_samples = batch_logistic_data.shape
        assert result.shape == (batch_size, n_samples)

    def test_batch_all_finite(self, batch_logistic_data):
        result = lyap_r_batch(batch_logistic_data, emb_dim=10, lag=2, trajectory_len=20, tau=1.0)
        assert jnp.all(jnp.isfinite(result))

    def test_batch_chaotic_positive(self, batch_logistic_data):
        result = lyap_r_batch(batch_logistic_data, emb_dim=10, lag=2, trajectory_len=20, tau=1.0)
        assert jnp.all(result > 0)


class TestLyapESingle:
    @pytest.fixture
    def logistic_map_data(self):
        def logistic_map(r: float, x0: float, n: int) -> jnp.ndarray:
            x = jnp.zeros(n)
            x = x.at[0].set(x0)
            for i in range(1, n):
                x = x.at[i].set(r * x[i - 1] * (1 - x[i - 1]))
            return x

        return logistic_map(3.9, 0.1, 2000)

    def test_output_shape(self, logistic_map_data):
        matrix_dim = 3
        result = lyap_e_single(
            logistic_map_data,
            emb_dim=10,
            matrix_dim=matrix_dim,
            min_nb=5,
            min_tsep=50,
            tau=1.0,
        )
        assert result.shape == (matrix_dim,)

    def test_all_finite(self, logistic_map_data):
        result = lyap_e_single(
            logistic_map_data, emb_dim=10, matrix_dim=3, min_nb=5, min_tsep=50, tau=1.0
        )
        assert jnp.all(jnp.isfinite(result))

    def test_exponents_ordered(self, logistic_map_data):
        result = lyap_e_single(
            logistic_map_data, emb_dim=10, matrix_dim=3, min_nb=5, min_tsep=50, tau=1.0
        )
        for i in range(len(result) - 1):
            assert result[i] >= result[i + 1]

    def test_largest_exponent_positive_for_chaos(self, logistic_map_data):
        result = lyap_e_single(
            logistic_map_data, emb_dim=10, matrix_dim=3, min_nb=5, min_tsep=50, tau=1.0
        )
        assert result[0] > 0


class TestLyapEBatch:
    @pytest.fixture
    def batch_logistic_data(self):
        def logistic_map(r: float, x0: float, n: int) -> np.ndarray:
            x = np.zeros(n)
            x[0] = x0
            for i in range(1, n):
                x[i] = r * x[i - 1] * (1 - x[i - 1])
            return x

        batch_size = 2
        n_samples = 2
        n_points = 1000
        data = np.zeros((n_points, batch_size, n_samples))
        for b in range(batch_size):
            for s in range(n_samples):
                r = 3.8 + 0.05 * b
                x0 = 0.1 + 0.01 * s
                data[:, b, s] = logistic_map(r, x0, n_points)

        return jnp.array(data)

    def test_batch_output_shape(self, batch_logistic_data):
        matrix_dim = 3
        result = lyap_e_batch(
            batch_logistic_data,
            emb_dim=10,
            matrix_dim=matrix_dim,
            min_nb=5,
            min_tsep=50,
            tau=1.0,
        )

        n_points, batch_size, n_samples = batch_logistic_data.shape
        assert result.shape == (batch_size, n_samples, matrix_dim)

    def test_batch_all_finite(self, batch_logistic_data):
        result = lyap_e_batch(
            batch_logistic_data,
            emb_dim=10,
            matrix_dim=3,
            min_nb=5,
            min_tsep=50,
            tau=1.0,
        )
        assert jnp.all(jnp.isfinite(result))


class TestLyapLogisticSign:
    """Test that lyap_r and lyap_e correctly identify stable vs chaotic regimes.

    Based on nolds TestNoldsLyap.test_lyap_logistic - tests sign of Lyapunov exponent
    for different r values in the logistic map:
    - r=2.5, 3.4: stable (negative exponent)
    - r=3.7, 4.0: chaotic (positive exponent)
    """

    @staticmethod
    def generate_logistic(r: float, x0: float, n: int) -> np.ndarray:
        x = np.zeros(n, dtype=np.float32)
        x[0] = x0
        for i in range(1, n):
            x[i] = r * x[i - 1] * (1 - x[i - 1])
        return x

    @pytest.mark.parametrize(
        "r,expected_sign",
        [
            (2.5, -1),  # stable fixed point
            (3.4, -1),  # stable period-2
            (3.7, 1),  # chaotic
            (4.0, 1),  # fully chaotic
        ],
    )
    def test_lyap_r_sign_detection(self, r, expected_sign):
        """Test that lyap_r correctly identifies sign for different r values."""
        data = self.generate_logistic(r, 0.1, 500)
        result = lyap_r_single(jnp.array(data), emb_dim=6, lag=2, trajectory_len=20, tau=1.0)
        assert int(np.sign(float(result))) == expected_sign, f"r={r}"

    @pytest.mark.parametrize(
        "r,expected_sign",
        [
            (2.5, -1),  # stable fixed point
            (3.4, -1),  # stable period-2
            (3.7, 1),  # chaotic
            (4.0, 1),  # fully chaotic
        ],
    )
    def test_lyap_e_sign_detection(self, r, expected_sign):
        """Test that lyap_e correctly identifies sign for different r values."""
        data = self.generate_logistic(r, 0.1, 500)
        result = lyap_e_single(
            jnp.array(data), emb_dim=6, matrix_dim=2, min_nb=5, min_tsep=10, tau=1.0
        )
        max_exponent = float(jnp.max(result))
        assert int(np.sign(max_exponent)) == expected_sign, f"r={r}"


class TestAccuracyVsNolds:
    @pytest.fixture
    def logistic_map_data(self):
        def logistic_map(r: float, x0: float, n: int) -> np.ndarray:
            x = np.zeros(n)
            x[0] = x0
            for i in range(1, n):
                x[i] = r * x[i - 1] * (1 - x[i - 1])
            return x

        return logistic_map(3.9, 0.1, 2000)

    def test_lyap_r_matches_nolds(self, logistic_map_data):
        nolds = pytest.importorskip("nolds")

        emb_dim = 10
        lag = 2
        trajectory_len = 20
        tau = 1.0

        jax_result = float(
            lyap_r_single(
                jnp.array(logistic_map_data),
                emb_dim=emb_dim,
                lag=lag,
                trajectory_len=trajectory_len,
                tau=tau,
            )
        )

        nolds_result = nolds.lyap_r(
            logistic_map_data,
            emb_dim=emb_dim,
            lag=lag,
            trajectory_len=trajectory_len,
            tau=tau,
            fit="poly",
        )

        np.testing.assert_allclose(jax_result, nolds_result, rtol=1e-5, atol=1e-6)

    def test_lyap_e_same_order_of_magnitude_as_nolds(self, logistic_map_data):
        nolds = pytest.importorskip("nolds")

        emb_dim = 7
        matrix_dim = 4
        min_nb = 5
        min_tsep = 50
        tau = 1.0

        jax_result = np.array(
            lyap_e_single(
                jnp.array(logistic_map_data),
                emb_dim=emb_dim,
                matrix_dim=matrix_dim,
                min_nb=min_nb,
                min_tsep=min_tsep,
                tau=tau,
            )
        )

        nolds_result = nolds.lyap_e(
            logistic_map_data,
            emb_dim=emb_dim,
            matrix_dim=matrix_dim,
            min_nb=min_nb,
            min_tsep=min_tsep,
            tau=tau,
        )

        assert len(jax_result) == len(nolds_result)
        assert np.sign(jax_result[0]) == np.sign(nolds_result[0])
        np.testing.assert_allclose(jax_result[0], nolds_result[0], rtol=0.5)


# =============================================================================
# Correlation Dimension Tests
# =============================================================================


class TestCorrDimSingle:
    """Tests for corr_dim_single function."""

    @pytest.fixture
    def linear_data(self):
        """Linear sequence - should have correlation dimension ~1."""
        return jnp.arange(1000, dtype=jnp.float32)

    @pytest.fixture
    def random_data(self):
        """Random data for testing."""
        np.random.seed(42)
        return jnp.array(np.random.random(1000).astype(np.float32))

    @pytest.fixture
    def logistic_map_data(self):
        """Chaotic logistic map data."""

        def logistic_map(r: float, x0: float, n: int) -> np.ndarray:
            x = np.zeros(n, dtype=np.float32)
            x[0] = x0
            for i in range(1, n):
                x[i] = r * x[i - 1] * (1 - x[i - 1])
            return x

        return jnp.array(logistic_map(3.9, 0.1, 1000))

    def test_linear_data_dimension_near_one(self, linear_data):
        """Test that linear data has correlation dimension close to 1."""
        result = corr_dim_single(linear_data, emb_dim=4, lag=1)
        assert jnp.isfinite(result)
        np.testing.assert_allclose(float(result), 1.0, atol=0.2)

    def test_output_is_scalar(self, random_data):
        """Test that output is a scalar."""
        result = corr_dim_single(random_data, emb_dim=4, lag=1)
        assert result.shape == ()

    def test_all_finite(self, logistic_map_data):
        """Test that result is finite."""
        result = corr_dim_single(logistic_map_data, emb_dim=4, lag=1)
        assert jnp.isfinite(result)

    def test_positive_result(self, logistic_map_data):
        """Test that correlation dimension is positive."""
        result = corr_dim_single(logistic_map_data, emb_dim=4, lag=1)
        assert float(result) > 0


class TestCorrDimBatch:
    """Tests for corr_dim_batch function."""

    @pytest.fixture
    def batch_data(self):
        """Generate batch of random data."""
        np.random.seed(42)
        batch_size = 4
        n_samples = 2
        n_points = 500
        data = np.random.random((n_points, batch_size, n_samples)).astype(np.float32)
        return jnp.array(data)

    @pytest.fixture
    def batch_logistic_data(self):
        """Generate batch of logistic map data."""

        def logistic_map(r: float, x0: float, n: int) -> np.ndarray:
            x = np.zeros(n, dtype=np.float32)
            x[0] = x0
            for i in range(1, n):
                x[i] = r * x[i - 1] * (1 - x[i - 1])
            return x

        batch_size = 3
        n_samples = 2
        n_points = 500
        data = np.zeros((n_points, batch_size, n_samples), dtype=np.float32)
        for b in range(batch_size):
            for s in range(n_samples):
                r = 3.8 + 0.05 * b
                x0 = 0.1 + 0.01 * s
                data[:, b, s] = logistic_map(r, x0, n_points)
        return jnp.array(data)

    def test_batch_output_shape(self, batch_data):
        """Test that output shape is correct."""
        result = corr_dim_batch(batch_data, emb_dim=4, lag=1)
        n_points, batch_size, n_samples = batch_data.shape
        assert result.shape == (batch_size, n_samples)

    def test_batch_all_finite(self, batch_logistic_data):
        """Test that all results are finite."""
        result = corr_dim_batch(batch_logistic_data, emb_dim=4, lag=1)
        assert jnp.all(jnp.isfinite(result))

    def test_batch_all_positive(self, batch_logistic_data):
        """Test that all correlation dimensions are positive."""
        result = corr_dim_batch(batch_logistic_data, emb_dim=4, lag=1)
        assert jnp.all(result > 0)


class TestCorrDimVsNolds:
    """Tests comparing JAX corr_dim to nolds implementation."""

    @pytest.fixture
    def logistic_map_data(self):
        """Logistic map data for comparison."""

        def logistic_map(r: float, x0: float, n: int) -> np.ndarray:
            x = np.zeros(n, dtype=np.float32)
            x[0] = x0
            for i in range(1, n):
                x[i] = r * x[i - 1] * (1 - x[i - 1])
            return x

        return logistic_map(3.9, 0.1, 1000)

    def test_same_order_of_magnitude_as_nolds(self, logistic_map_data):
        """Test that JAX result is in same ballpark as nolds."""
        nolds = pytest.importorskip("nolds")

        emb_dim = 4
        lag = 1

        jax_result = float(corr_dim_single(jnp.array(logistic_map_data), emb_dim=emb_dim, lag=lag))
        nolds_result = nolds.corr_dim(logistic_map_data, emb_dim=emb_dim, lag=lag, fit="poly")

        np.testing.assert_allclose(jax_result, nolds_result, rtol=0.2)

    def test_linear_data_matches_nolds(self):
        """Test that linear data gives similar results to nolds."""
        nolds = pytest.importorskip("nolds")

        data = np.arange(1000, dtype=np.float32)
        emb_dim = 4
        lag = 1

        jax_result = float(corr_dim_single(jnp.array(data), emb_dim=emb_dim, lag=lag))
        nolds_result = nolds.corr_dim(data, emb_dim=emb_dim, lag=lag, fit="poly")

        np.testing.assert_allclose(jax_result, nolds_result, atol=0.15)
