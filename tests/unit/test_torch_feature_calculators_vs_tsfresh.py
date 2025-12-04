# pyright: basic
"""Tests comparing PyTorch feature calculators against tsfresh implementations.

This test suite verifies that our PyTorch implementations produce results
consistent with tsfresh's reference implementations.
"""

import numpy as np
import pytest
import torch
from tsfresh.feature_extraction import feature_calculators as fc

from pybasin.feature_extractors import torch_feature_calculators as torch_fc

# Tolerance for floating point comparisons
RTOL = 1e-4  # Relative tolerance
ATOL = 1e-6  # Absolute tolerance

# Looser tolerance for complex calculations
RTOL_LOOSE = 0.1
ATOL_LOOSE = 0.1


@pytest.fixture
def sample_data():
    """Generate sample time series data for testing."""
    np.random.seed(42)
    return np.random.randn(100)


@pytest.fixture
def sample_data_torch(sample_data):
    """Convert sample data to PyTorch format (N, B, S) = (100, 1, 1)."""
    return torch.from_numpy(sample_data.reshape(-1, 1, 1).astype(np.float32))


@pytest.fixture
def positive_data():
    """Generate positive sample data."""
    np.random.seed(42)
    return np.abs(np.random.randn(100)) + 0.1


@pytest.fixture
def positive_data_torch(positive_data):
    """Convert positive data to PyTorch format."""
    return torch.from_numpy(positive_data.reshape(-1, 1, 1).astype(np.float32))


@pytest.fixture
def integer_data():
    """Generate integer data with duplicates."""
    np.random.seed(42)
    return np.random.randint(0, 10, size=100).astype(float)


@pytest.fixture
def integer_data_torch(integer_data):
    """Convert integer data to PyTorch format."""
    return torch.from_numpy(integer_data.reshape(-1, 1, 1).astype(np.float32))


# =============================================================================
# MINIMAL FEATURES
# =============================================================================


class TestMinimalFeatures:
    """Tests for tsfresh MinimalFCParameters features."""

    def test_sum_values(self, sample_data, sample_data_torch):
        expected = fc.sum_values(sample_data)
        result = float(torch_fc.sum_values(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_median(self, sample_data, sample_data_torch):
        # Note: torch.median returns lower middle value for even n, numpy averages
        # For odd n they match; for even n we verify torch result is the lower middle value
        result = float(torch_fc.median(sample_data_torch)[0, 0])
        sorted_data = np.sort(sample_data)
        n = len(sample_data)
        if n % 2 == 0:
            # For even n, torch returns lower middle value
            expected_lower = sorted_data[n // 2 - 1]
            assert np.isclose(result, expected_lower, rtol=RTOL, atol=ATOL)
        else:
            expected = fc.median(sample_data)
            assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_mean(self, sample_data, sample_data_torch):
        expected = fc.mean(sample_data)
        result = float(torch_fc.mean(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_length(self, sample_data, sample_data_torch):
        expected = fc.length(sample_data)
        result = float(torch_fc.length(sample_data_torch)[0, 0])
        assert result == expected

    def test_standard_deviation(self, sample_data, sample_data_torch):
        expected = fc.standard_deviation(sample_data)
        result = float(torch_fc.standard_deviation(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_variance(self, sample_data, sample_data_torch):
        expected = fc.variance(sample_data)
        result = float(torch_fc.variance(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_root_mean_square(self, sample_data, sample_data_torch):
        expected = fc.root_mean_square(sample_data)
        result = float(torch_fc.root_mean_square(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_maximum(self, sample_data, sample_data_torch):
        expected = fc.maximum(sample_data)
        result = float(torch_fc.maximum(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_absolute_maximum(self, sample_data, sample_data_torch):
        expected = fc.absolute_maximum(sample_data)
        result = float(torch_fc.absolute_maximum(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_minimum(self, sample_data, sample_data_torch):
        expected = fc.minimum(sample_data)
        result = float(torch_fc.minimum(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)


# =============================================================================
# SIMPLE STATISTICS
# =============================================================================


class TestSimpleStatistics:
    """Tests for simple statistics features."""

    def test_abs_energy(self, sample_data, sample_data_torch):
        expected = fc.abs_energy(sample_data)
        result = float(torch_fc.abs_energy(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_kurtosis(self, sample_data, sample_data_torch):
        expected = fc.kurtosis(sample_data)
        result = float(torch_fc.kurtosis(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=0.1, atol=0.5)

    def test_skewness(self, sample_data, sample_data_torch):
        expected = fc.skewness(sample_data)
        result = float(torch_fc.skewness(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=0.1, atol=0.1)

    def test_quantile(self, sample_data, sample_data_torch):
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            expected = fc.quantile(sample_data, q)
            result = float(torch_fc.quantile(sample_data_torch, q)[0, 0])
            assert np.isclose(result, expected, rtol=RTOL, atol=ATOL), f"Failed for q={q}"

    def test_variation_coefficient(self, positive_data, positive_data_torch):
        expected = fc.variation_coefficient(positive_data)
        result = float(torch_fc.variation_coefficient(positive_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=0.01, atol=ATOL)


# =============================================================================
# CHANGE/DIFFERENCE FEATURES
# =============================================================================


class TestChangeFeatures:
    """Tests for change/difference based features."""

    def test_absolute_sum_of_changes(self, sample_data, sample_data_torch):
        expected = fc.absolute_sum_of_changes(sample_data)
        result = float(torch_fc.absolute_sum_of_changes(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_mean_abs_change(self, sample_data, sample_data_torch):
        expected = fc.mean_abs_change(sample_data)
        result = float(torch_fc.mean_abs_change(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_mean_change(self, sample_data, sample_data_torch):
        expected = fc.mean_change(sample_data)
        result = float(torch_fc.mean_change(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)


# =============================================================================
# COUNTING FEATURES
# =============================================================================


class TestCountingFeatures:
    """Tests for counting features."""

    def test_count_above(self, sample_data, sample_data_torch):
        for t in [-1.0, 0.0, 0.5, 1.0]:
            expected = fc.count_above(sample_data, t)
            result = float(torch_fc.count_above(sample_data_torch, t)[0, 0])
            assert np.isclose(result, expected, rtol=RTOL, atol=ATOL), f"Failed for t={t}"

    def test_count_above_mean(self, sample_data, sample_data_torch):
        expected = fc.count_above_mean(sample_data)
        result = float(torch_fc.count_above_mean(sample_data_torch)[0, 0])
        assert result == expected

    def test_count_below(self, sample_data, sample_data_torch):
        for t in [-1.0, 0.0, 0.5, 1.0]:
            expected = fc.count_below(sample_data, t)
            result = float(torch_fc.count_below(sample_data_torch, t)[0, 0])
            assert np.isclose(result, expected, rtol=RTOL, atol=ATOL), f"Failed for t={t}"

    def test_count_below_mean(self, sample_data, sample_data_torch):
        expected = fc.count_below_mean(sample_data)
        result = float(torch_fc.count_below_mean(sample_data_torch)[0, 0])
        assert result == expected


# =============================================================================
# BOOLEAN FEATURES
# =============================================================================


class TestBooleanFeatures:
    """Tests for boolean features."""

    def test_has_duplicate(self, integer_data, integer_data_torch):
        expected = fc.has_duplicate(integer_data)
        result = bool(torch_fc.has_duplicate(integer_data_torch)[0, 0])
        assert result == expected

    def test_has_duplicate_max(self, integer_data, integer_data_torch):
        expected = fc.has_duplicate_max(integer_data)
        result = bool(torch_fc.has_duplicate_max(integer_data_torch)[0, 0])
        assert result == expected

    def test_has_duplicate_min(self, integer_data, integer_data_torch):
        expected = fc.has_duplicate_min(integer_data)
        result = bool(torch_fc.has_duplicate_min(integer_data_torch)[0, 0])
        assert result == expected

    def test_large_standard_deviation(self, sample_data, sample_data_torch):
        for r in [0.1, 0.25, 0.5]:
            expected = fc.large_standard_deviation(sample_data, r)
            result = bool(torch_fc.has_large_standard_deviation(sample_data_torch, r)[0, 0])
            assert result == expected, f"Failed for r={r}"


# =============================================================================
# LOCATION FEATURES
# =============================================================================


class TestLocationFeatures:
    """Tests for location features."""

    def test_first_location_of_maximum(self, sample_data, sample_data_torch):
        expected = fc.first_location_of_maximum(sample_data)
        result = float(torch_fc.first_location_of_maximum(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_first_location_of_minimum(self, sample_data, sample_data_torch):
        expected = fc.first_location_of_minimum(sample_data)
        result = float(torch_fc.first_location_of_minimum(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_last_location_of_maximum(self, sample_data, sample_data_torch):
        expected = fc.last_location_of_maximum(sample_data)
        result = float(torch_fc.last_location_of_maximum(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=0.02, atol=0.02)

    def test_last_location_of_minimum(self, sample_data, sample_data_torch):
        expected = fc.last_location_of_minimum(sample_data)
        result = float(torch_fc.last_location_of_minimum(sample_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=0.02, atol=0.02)


# =============================================================================
# STREAK/PATTERN FEATURES
# =============================================================================


class TestStreakFeatures:
    """Tests for streak/pattern features."""

    def test_longest_strike_above_mean(self, sample_data, sample_data_torch):
        expected = fc.longest_strike_above_mean(sample_data)
        result = float(torch_fc.longest_strike_above_mean(sample_data_torch)[0, 0])
        assert result == expected

    def test_longest_strike_below_mean(self, sample_data, sample_data_torch):
        expected = fc.longest_strike_below_mean(sample_data)
        result = float(torch_fc.longest_strike_below_mean(sample_data_torch)[0, 0])
        assert result == expected

    def test_number_crossing_m(self, sample_data, sample_data_torch):
        for m in [-0.5, 0.0, 0.5]:
            expected = fc.number_crossing_m(sample_data, m)
            result = float(torch_fc.number_crossing_m(sample_data_torch, m)[0, 0])
            assert result == expected, f"Failed for m={m}"

    def test_number_peaks(self, sample_data, sample_data_torch):
        for n in [1, 2, 3]:
            expected = fc.number_peaks(sample_data, n)
            result = float(torch_fc.number_peaks(sample_data_torch, n)[0, 0])
            assert result == expected, f"Failed for n={n}"


# =============================================================================
# AUTOCORRELATION FEATURES
# =============================================================================


class TestAutocorrelationFeatures:
    """Tests for autocorrelation features."""

    def test_autocorrelation(self, sample_data, sample_data_torch):
        for lag in [1, 5, 10]:
            expected = fc.autocorrelation(sample_data, lag)
            result = float(torch_fc.autocorrelation(sample_data_torch, lag)[0, 0])
            assert np.isclose(result, expected, rtol=0.01, atol=0.01), f"Failed for lag={lag}"


# =============================================================================
# ENTROPY/COMPLEXITY FEATURES
# =============================================================================


class TestEntropyFeatures:
    """Tests for entropy/complexity features."""

    def test_binned_entropy(self, sample_data, sample_data_torch):
        expected = fc.binned_entropy(sample_data, 10)
        result = float(torch_fc.binned_entropy(sample_data_torch, 10)[0, 0])
        assert np.isclose(result, expected, rtol=0.1, atol=0.1)

    def test_cid_ce(self, sample_data, sample_data_torch):
        expected = fc.cid_ce(sample_data, normalize=True)
        result = float(torch_fc.cid_ce(sample_data_torch, normalize=True)[0, 0])
        assert np.isclose(result, expected, rtol=0.01, atol=0.01)


# =============================================================================
# FREQUENCY DOMAIN FEATURES
# =============================================================================


class TestFrequencyFeatures:
    """Tests for frequency domain features."""

    def test_fft_coefficient_abs(self, sample_data, sample_data_torch):
        result_list = list(fc.fft_coefficient(sample_data, [{"coeff": 0, "attr": "abs"}]))
        expected = result_list[0][1] if result_list else 0.0
        result = float(torch_fc.fft_coefficient(sample_data_torch, 0, "abs")[0, 0])
        assert np.isclose(result, expected, rtol=0.01, atol=0.01)


# =============================================================================
# TREND/REGRESSION FEATURES
# =============================================================================


class TestTrendFeatures:
    """Tests for trend/regression features."""

    def test_linear_trend_slope(self, sample_data, sample_data_torch):
        expected = fc.linear_trend(sample_data, [{"attr": "slope"}])[0][1]
        result = float(torch_fc.linear_trend(sample_data_torch, "slope")[0, 0])
        assert np.isclose(result, expected, rtol=0.01, atol=0.001)

    def test_linear_trend_intercept(self, sample_data, sample_data_torch):
        expected = fc.linear_trend(sample_data, [{"attr": "intercept"}])[0][1]
        result = float(torch_fc.linear_trend(sample_data_torch, "intercept")[0, 0])
        assert np.isclose(result, expected, rtol=0.01, atol=0.01)


# =============================================================================
# REOCCURRENCE FEATURES
# =============================================================================


class TestReoccurrenceFeatures:
    """Tests for reoccurrence features."""

    def test_ratio_value_number_to_time_series_length(self, integer_data, integer_data_torch):
        expected = fc.ratio_value_number_to_time_series_length(integer_data)
        result = float(torch_fc.ratio_value_number_to_time_series_length(integer_data_torch)[0, 0])
        assert np.isclose(result, expected, rtol=RTOL, atol=ATOL)


# =============================================================================
# ADVANCED FEATURES
# =============================================================================


class TestAdvancedFeatures:
    """Tests for advanced features."""

    def test_c3(self, sample_data, sample_data_torch):
        for lag in [1, 2, 3]:
            expected = fc.c3(sample_data, lag)
            result = float(torch_fc.c3(sample_data_torch, lag)[0, 0])
            assert np.isclose(result, expected, rtol=0.01, atol=0.01), f"Failed for lag={lag}"

    def test_mean_n_absolute_max(self, sample_data, sample_data_torch):
        for n in [1, 3, 5]:
            expected = fc.mean_n_absolute_max(sample_data, n)
            result = float(torch_fc.mean_n_absolute_max(sample_data_torch, n)[0, 0])
            assert np.isclose(result, expected, rtol=0.01, atol=0.01), f"Failed for n={n}"

    def test_range_count(self, sample_data, sample_data_torch):
        expected = fc.range_count(sample_data, -1.0, 1.0)
        result = float(torch_fc.range_count(sample_data_torch, -1.0, 1.0)[0, 0])
        assert result == expected

    def test_ratio_beyond_r_sigma(self, sample_data, sample_data_torch):
        for r in [1.0, 2.0, 3.0]:
            expected = fc.ratio_beyond_r_sigma(sample_data, r)
            result = float(torch_fc.ratio_beyond_r_sigma(sample_data_torch, r)[0, 0])
            assert np.isclose(result, expected, rtol=0.01, atol=0.01), f"Failed for r={r}"

    def test_symmetry_looking(self, sample_data, sample_data_torch):
        for r in [0.05, 0.1, 0.25]:
            expected = fc.symmetry_looking(sample_data, [{"r": r}])[0][1]
            result = bool(torch_fc.symmetry_looking(sample_data_torch, r)[0, 0])
            assert result == expected, f"Failed for r={r}"

    def test_value_count(self, integer_data, integer_data_torch):
        for value in [0.0, 1.0, 5.0]:
            expected = fc.value_count(integer_data, value)
            result = float(torch_fc.value_count(integer_data_torch, value)[0, 0])
            assert result == expected, f"Failed for value={value}"


# =============================================================================
# BATCH PROCESSING TESTS
# =============================================================================


class TestBatchProcessing:
    """Tests verifying batch processing works correctly."""

    def test_batch_mean(self):
        """Test that batch processing produces correct results for each trajectory."""
        np.random.seed(42)
        data1 = np.random.randn(100)
        data2 = np.random.randn(100) + 5
        data3 = np.random.randn(100) - 3

        data_torch = torch.from_numpy(
            np.stack([data1, data2, data3], axis=1).reshape(100, 3, 1).astype(np.float32)
        )

        result = torch_fc.mean(data_torch)

        assert np.isclose(float(result[0, 0]), np.mean(data1), rtol=RTOL, atol=ATOL)
        assert np.isclose(float(result[1, 0]), np.mean(data2), rtol=RTOL, atol=ATOL)
        assert np.isclose(float(result[2, 0]), np.mean(data3), rtol=RTOL, atol=ATOL)

    def test_multi_state_processing(self):
        """Test that multiple state variables are processed correctly."""
        np.random.seed(42)
        state1 = np.random.randn(100)
        state2 = np.random.randn(100) * 2

        data_torch = torch.from_numpy(
            np.stack([state1, state2], axis=1).reshape(100, 1, 2).astype(np.float32)
        )

        result = torch_fc.standard_deviation(data_torch)

        assert np.isclose(float(result[0, 0]), np.std(state1), rtol=RTOL, atol=ATOL)
        assert np.isclose(float(result[0, 1]), np.std(state2), rtol=RTOL, atol=ATOL)


# =============================================================================
# EXTRACT FEATURES FROM CONFIG TESTS
# =============================================================================


class TestExtractFeaturesFromConfig:
    """Tests for the extract_features_from_config function."""

    def test_default_config_returns_features(self, sample_data_torch):
        """Test that default config extracts features."""
        features = torch_fc.extract_features_from_config(sample_data_torch)
        assert len(features) > 0
        assert "mean" in features
        assert "variance" in features

    def test_custom_config(self, sample_data_torch):
        """Test that custom config works."""
        config = {
            "mean": None,
            "variance": None,
            "autocorrelation": [{"lag": 1}, {"lag": 2}],
        }
        features = torch_fc.extract_features_from_config(sample_data_torch, config)
        assert "mean" in features
        assert "variance" in features
        assert "autocorrelation__lag_1" in features
        assert "autocorrelation__lag_2" in features
        assert len(features) == 4

    def test_include_custom_features(self, sample_data_torch):
        """Test that custom features can be included."""
        config = {"mean": None}
        features = torch_fc.extract_features_from_config(
            sample_data_torch, config, include_custom=True
        )
        assert "mean" in features
        assert "delta" in features
        assert "log_delta" in features

    def test_feature_shape(self, sample_data_torch):
        """Test that extracted features have correct shape."""
        features = torch_fc.extract_features_from_config(sample_data_torch, {"mean": None})
        assert features["mean"].shape == (1, 1)


class TestComprehensiveFeatureSet:
    """Tests verifying PyTorch config structure."""

    def test_all_config_features_exist_in_functions(self):
        """Verify all features in config exist in ALL_FEATURE_FUNCTIONS."""
        for feature_name in torch_fc.TORCH_COMPREHENSIVE_FC_PARAMETERS:
            assert feature_name in torch_fc.ALL_FEATURE_FUNCTIONS, f"Missing: {feature_name}"

    def test_custom_features_separate(self):
        """Verify custom features are in separate config."""
        assert "delta" in torch_fc.TORCH_CUSTOM_FC_PARAMETERS
        assert "log_delta" in torch_fc.TORCH_CUSTOM_FC_PARAMETERS

    def test_comprehensive_feature_count(self):
        """Verify we have a reasonable number of features."""
        names = torch_fc.get_feature_names_from_config()
        assert len(names) > 100
