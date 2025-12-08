# pyright: basic
"""Tests for batched PyTorch feature calculators.

This test suite verifies that batched implementations produce results
consistent with their loop-based counterparts.
"""

import pytest
import torch

from pybasin.feature_extractors import torch_feature_calculators as torch_fc
from pybasin.feature_extractors.torch_batched_calculators import (
    agg_autocorrelation_batched,
    agg_linear_trend_batched,
    augmented_dickey_fuller_batched,
    c3_batched,
    change_quantiles_batched,
    fft_aggregated_batched,
    fourier_entropy_batched,
    friedrich_coefficients_batched,
    mean_n_absolute_max_batched,
    number_crossing_m_batched,
    number_peaks_batched,
    partial_autocorrelation_batched,
    range_count_batched,
    spkt_welch_density_batched,
    time_reversal_asymmetry_statistic_batched,
    value_count_batched,
)

RTOL = 1e-5
ATOL = 1e-6


@pytest.fixture
def sample_data_torch():
    """Generate sample time series data in PyTorch format (N, B, S)."""
    torch.manual_seed(42)
    return torch.randn(100, 1, 1, dtype=torch.float32)


@pytest.fixture
def sample_data_torch_batched():
    """Generate batched sample data (N, B, S) with multiple batches/states."""
    torch.manual_seed(42)
    return torch.randn(100, 4, 3, dtype=torch.float32)


class TestChangeQuantilesBatched:
    """Tests for change_quantiles_batched function."""

    def test_single_param_matches_loop(self, sample_data_torch):
        """Single parameter should match non-batched version."""
        params = [{"ql": 0.0, "qh": 0.2, "isabs": True, "f_agg": "mean"}]
        batched_result = change_quantiles_batched(sample_data_torch, params)
        loop_result = torch_fc.change_quantiles(
            sample_data_torch, ql=0.0, qh=0.2, isabs=True, f_agg="mean"
        )
        assert torch.allclose(batched_result[0], loop_result, rtol=RTOL, atol=ATOL)

    def test_multiple_params_match_loop(self, sample_data_torch):
        """Multiple parameters should match loop over non-batched calls."""
        params = [
            {"ql": 0.0, "qh": 0.2, "isabs": True, "f_agg": "mean"},
            {"ql": 0.0, "qh": 0.2, "isabs": True, "f_agg": "var"},
            {"ql": 0.0, "qh": 0.2, "isabs": False, "f_agg": "mean"},
            {"ql": 0.2, "qh": 0.8, "isabs": True, "f_agg": "mean"},
            {"ql": 0.4, "qh": 1.0, "isabs": False, "f_agg": "var"},
        ]
        batched_result = change_quantiles_batched(sample_data_torch, params)

        for idx, p in enumerate(params):
            loop_result = torch_fc.change_quantiles(
                sample_data_torch, ql=p["ql"], qh=p["qh"], isabs=p["isabs"], f_agg=p["f_agg"]
            )
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at param {idx}: {p}"
            )

    def test_empty_params(self, sample_data_torch):
        """Empty params should return empty tensor."""
        result = change_quantiles_batched(sample_data_torch, [])
        assert result.shape == (0, 1, 1)

    def test_batched_data(self, sample_data_torch_batched):
        """Should work with batched data (B > 1, S > 1)."""
        params = [
            {"ql": 0.0, "qh": 0.5, "isabs": True, "f_agg": "mean"},
            {"ql": 0.2, "qh": 0.8, "isabs": False, "f_agg": "var"},
        ]
        batched_result = change_quantiles_batched(sample_data_torch_batched, params)
        assert batched_result.shape == (2, 4, 3)

        for idx, p in enumerate(params):
            loop_result = torch_fc.change_quantiles(
                sample_data_torch_batched,
                ql=p["ql"],
                qh=p["qh"],
                isabs=p["isabs"],
                f_agg=p["f_agg"],
            )
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL)

    def test_full_80_params(self, sample_data_torch):
        """Test with all 80 tsfresh parameter combinations."""
        params = []
        for ql in [0.0, 0.2, 0.4, 0.6, 0.8]:
            for qh in [0.2, 0.4, 0.6, 0.8, 1.0]:
                if ql < qh:
                    for isabs in [True, False]:
                        for f_agg in ["mean", "var"]:
                            params.append({"ql": ql, "qh": qh, "isabs": isabs, "f_agg": f_agg})

        batched_result = change_quantiles_batched(sample_data_torch, params)
        assert batched_result.shape[0] == len(params)

        for idx in range(0, len(params), 10):
            p = params[idx]
            loop_result = torch_fc.change_quantiles(
                sample_data_torch, ql=p["ql"], qh=p["qh"], isabs=p["isabs"], f_agg=p["f_agg"]
            )
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL)


class TestAggLinearTrendBatched:
    """Tests for agg_linear_trend_batched function."""

    def test_single_param_matches_loop(self, sample_data_torch):
        """Single parameter should match non-batched version."""
        params = [{"chunk_size": 10, "f_agg": "mean", "attr": "slope"}]
        batched_result = agg_linear_trend_batched(sample_data_torch, params)
        loop_result = torch_fc.agg_linear_trend(
            sample_data_torch, chunk_size=10, f_agg="mean", attr="slope"
        )
        assert torch.allclose(batched_result[0], loop_result, rtol=RTOL, atol=ATOL)

    def test_multiple_params_match_loop(self, sample_data_torch):
        """Multiple parameters should match loop over non-batched calls."""
        params = [
            {"chunk_size": 5, "f_agg": "mean", "attr": "slope"},
            {"chunk_size": 5, "f_agg": "mean", "attr": "intercept"},
            {"chunk_size": 5, "f_agg": "var", "attr": "rvalue"},
            {"chunk_size": 10, "f_agg": "max", "attr": "stderr"},
        ]
        batched_result = agg_linear_trend_batched(sample_data_torch, params)

        for idx, p in enumerate(params):
            loop_result = torch_fc.agg_linear_trend(
                sample_data_torch,
                chunk_size=p["chunk_size"],
                f_agg=p["f_agg"],
                attr=p["attr"],
            )
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at param {idx}: {p}"
            )

    def test_empty_params(self, sample_data_torch):
        """Empty params should return empty tensor."""
        result = agg_linear_trend_batched(sample_data_torch, [])
        assert result.shape == (0, 1, 1)

    def test_batched_data(self, sample_data_torch_batched):
        """Should work with batched data."""
        params = [
            {"chunk_size": 10, "f_agg": "mean", "attr": "slope"},
            {"chunk_size": 10, "f_agg": "var", "attr": "intercept"},
        ]
        batched_result = agg_linear_trend_batched(sample_data_torch_batched, params)
        assert batched_result.shape == (2, 4, 3)

    def test_all_48_params(self, sample_data_torch):
        """Test with all 48 tsfresh parameter combinations."""
        params = []
        for chunk_size in [5, 10, 50]:
            for f_agg in ["mean", "var", "min", "max"]:
                for attr in ["slope", "intercept", "rvalue", "stderr"]:
                    params.append({"chunk_size": chunk_size, "f_agg": f_agg, "attr": attr})

        batched_result = agg_linear_trend_batched(sample_data_torch, params)
        assert batched_result.shape[0] == len(params)

        for idx, p in enumerate(params):
            loop_result = torch_fc.agg_linear_trend(
                sample_data_torch,
                chunk_size=p["chunk_size"],
                f_agg=p["f_agg"],
                attr=p["attr"],
            )
            both_nan = torch.isnan(batched_result[idx]) & torch.isnan(loop_result)
            close_or_nan = (
                torch.allclose(
                    batched_result[idx], loop_result, rtol=RTOL, atol=ATOL, equal_nan=True
                )
                or both_nan.all()
            )
            assert close_or_nan, f"Mismatch at param {idx}: {p}"


class TestPartialAutocorrelationBatched:
    """Tests for partial_autocorrelation_batched function."""

    def test_single_lag_matches_loop(self, sample_data_torch):
        """Single lag should match non-batched version."""
        lags = [5]
        batched_result = partial_autocorrelation_batched(sample_data_torch, lags)
        loop_result = torch_fc.partial_autocorrelation(sample_data_torch, lag=5)
        assert torch.allclose(batched_result[0], loop_result, rtol=RTOL, atol=ATOL)

    def test_multiple_lags_match_loop(self, sample_data_torch):
        """Multiple lags should match loop over non-batched calls."""
        lags = [0, 1, 2, 5, 10]
        batched_result = partial_autocorrelation_batched(sample_data_torch, lags)

        for idx, lag in enumerate(lags):
            loop_result = torch_fc.partial_autocorrelation(sample_data_torch, lag=lag)
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at lag {lag}"
            )

    def test_empty_lags(self, sample_data_torch):
        """Empty lags should return empty tensor."""
        result = partial_autocorrelation_batched(sample_data_torch, [])
        assert result.shape == (0, 1, 1)

    def test_lag_zero(self, sample_data_torch):
        """Lag 0 should return 1.0."""
        result = partial_autocorrelation_batched(sample_data_torch, [0])
        assert torch.allclose(result[0], torch.ones(1, 1))

    def test_batched_data(self, sample_data_torch_batched):
        """Should work with batched data."""
        lags = [1, 2, 3]
        batched_result = partial_autocorrelation_batched(sample_data_torch_batched, lags)
        assert batched_result.shape == (3, 4, 3)

        for idx, lag in enumerate(lags):
            loop_result = torch_fc.partial_autocorrelation(sample_data_torch_batched, lag=lag)
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL)

    def test_all_10_lags(self, sample_data_torch):
        """Test with typical tsfresh lags [0..9]."""
        lags = list(range(10))
        batched_result = partial_autocorrelation_batched(sample_data_torch, lags)
        assert batched_result.shape[0] == 10

        for idx, lag in enumerate(lags):
            loop_result = torch_fc.partial_autocorrelation(sample_data_torch, lag=lag)
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at lag {lag}"
            )

    def test_large_lag_beyond_data(self, sample_data_torch):
        """Lags beyond data length should return zeros."""
        lags = [50, 100, 150]
        batched_result = partial_autocorrelation_batched(sample_data_torch, lags)
        for idx, lag in enumerate(lags):
            if lag >= sample_data_torch.shape[0]:
                assert torch.allclose(batched_result[idx], torch.zeros(1, 1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBatchedCalculatorsGPU:
    """GPU-specific tests for batched calculators."""

    def test_change_quantiles_gpu(self):
        """Test change_quantiles_batched on GPU."""
        x = torch.randn(100, 2, 2, dtype=torch.float32, device="cuda")
        params = [
            {"ql": 0.0, "qh": 0.5, "isabs": True, "f_agg": "mean"},
            {"ql": 0.2, "qh": 0.8, "isabs": False, "f_agg": "var"},
        ]
        result = change_quantiles_batched(x, params)
        assert result.device.type == "cuda"
        assert result.shape == (2, 2, 2)

    def test_agg_linear_trend_gpu(self):
        """Test agg_linear_trend_batched on GPU."""
        x = torch.randn(100, 2, 2, dtype=torch.float32, device="cuda")
        params = [
            {"chunk_size": 10, "f_agg": "mean", "attr": "slope"},
            {"chunk_size": 10, "f_agg": "var", "attr": "intercept"},
        ]
        result = agg_linear_trend_batched(x, params)
        assert result.device.type == "cuda"
        assert result.shape == (2, 2, 2)

    def test_partial_autocorrelation_gpu(self):
        """Test partial_autocorrelation_batched on GPU."""
        x = torch.randn(100, 2, 2, dtype=torch.float32, device="cuda")
        lags = [1, 5, 10]
        result = partial_autocorrelation_batched(x, lags)
        assert result.device.type == "cuda"
        assert result.shape == (3, 2, 2)


# =============================================================================
# PHASE 2: MEDIUM PRIORITY BATCHED FUNCTION TESTS
# =============================================================================


class TestFourierEntropyBatched:
    """Tests for fourier_entropy_batched function."""

    def test_single_bins_matches_loop(self, sample_data_torch):
        """Single bins value should match non-batched version."""
        bins_list = [10]
        batched_result = fourier_entropy_batched(sample_data_torch, bins_list)
        loop_result = torch_fc.fourier_entropy(sample_data_torch, bins=10)
        assert torch.allclose(batched_result[0], loop_result, rtol=RTOL, atol=ATOL)

    def test_multiple_bins_match_loop(self, sample_data_torch):
        """Multiple bins values should match loop over non-batched calls."""
        bins_list = [2, 5, 10, 20, 50]
        batched_result = fourier_entropy_batched(sample_data_torch, bins_list)

        for idx, bins in enumerate(bins_list):
            loop_result = torch_fc.fourier_entropy(sample_data_torch, bins=bins)
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at bins {bins}"
            )

    def test_empty_bins_list(self, sample_data_torch):
        """Empty bins list should return empty tensor."""
        result = fourier_entropy_batched(sample_data_torch, [])
        assert result.shape == (0, 1, 1)


class TestFftAggregatedBatched:
    """Tests for fft_aggregated_batched function."""

    def test_single_aggtype_matches_loop(self, sample_data_torch):
        """Single aggtype should match non-batched version."""
        aggtypes = ["centroid"]
        batched_result = fft_aggregated_batched(sample_data_torch, aggtypes)
        loop_result = torch_fc.fft_aggregated(sample_data_torch, aggtype="centroid")
        assert torch.allclose(batched_result[0], loop_result, rtol=RTOL, atol=ATOL)

    def test_all_aggtypes_match_loop(self, sample_data_torch):
        """All 4 aggtypes should match loop over non-batched calls."""
        aggtypes = ["centroid", "variance", "skew", "kurtosis"]
        batched_result = fft_aggregated_batched(sample_data_torch, aggtypes)

        for idx, aggtype in enumerate(aggtypes):
            loop_result = torch_fc.fft_aggregated(sample_data_torch, aggtype=aggtype)
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at aggtype {aggtype}"
            )

    def test_empty_aggtypes(self, sample_data_torch):
        """Empty aggtypes should raise ValueError."""
        with pytest.raises(ValueError, match="aggtypes cannot be empty"):
            fft_aggregated_batched(sample_data_torch, [])

    def test_batched_data(self, sample_data_torch_batched):
        """Should work with batched data."""
        aggtypes = ["centroid", "variance"]
        batched_result = fft_aggregated_batched(sample_data_torch_batched, aggtypes)
        assert batched_result.shape == (2, 4, 3)


class TestSpktWelchDensityBatched:
    """Tests for spkt_welch_density_batched function."""

    def test_single_coeff_matches_loop(self, sample_data_torch):
        """Single coeff should match non-batched version."""
        coeffs = [0]
        batched_result = spkt_welch_density_batched(sample_data_torch, coeffs)
        loop_result = torch_fc.spkt_welch_density(sample_data_torch, coeff=0)
        assert torch.allclose(batched_result[0], loop_result, rtol=RTOL, atol=ATOL)

    def test_multiple_coeffs_match_loop(self, sample_data_torch):
        """Multiple coeffs should match loop over non-batched calls."""
        coeffs = [0, 1, 2]
        batched_result = spkt_welch_density_batched(sample_data_torch, coeffs)

        for idx, coeff in enumerate(coeffs):
            loop_result = torch_fc.spkt_welch_density(sample_data_torch, coeff=coeff)
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at coeff {coeff}"
            )

    def test_coeff_beyond_size(self, sample_data_torch):
        """Coeff beyond FFT size should return zeros."""
        coeffs = [0, 100]
        batched_result = spkt_welch_density_batched(sample_data_torch, coeffs)
        assert torch.allclose(batched_result[1], torch.zeros(1, 1))


class TestNumberPeaksBatched:
    """Tests for number_peaks_batched function."""

    def test_single_n_matches_loop(self, sample_data_torch):
        """Single n should match non-batched version."""
        ns = [3]
        batched_result = number_peaks_batched(sample_data_torch, ns)
        loop_result = torch_fc.number_peaks(sample_data_torch, n=3)
        assert torch.allclose(batched_result[0], loop_result, rtol=RTOL, atol=ATOL)

    def test_multiple_ns_match_loop(self, sample_data_torch):
        """Multiple n values should match loop over non-batched calls."""
        ns = [1, 3, 5, 10, 20]
        batched_result = number_peaks_batched(sample_data_torch, ns)

        for idx, n in enumerate(ns):
            loop_result = torch_fc.number_peaks(sample_data_torch, n=n)
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at n {n}"
            )

    def test_n_too_large(self, sample_data_torch):
        """n too large for data should return zeros."""
        ns = [1, 60]
        batched_result = number_peaks_batched(sample_data_torch, ns)
        assert torch.allclose(batched_result[1], torch.zeros(1, 1))


class TestFriedrichCoefficientsBatched:
    """Tests for friedrich_coefficients_batched function."""

    def test_single_param_matches_loop(self, sample_data_torch):
        """Single param should match non-batched version."""
        params = [{"m": 3, "r": 30.0, "coeff": 0}]
        batched_result = friedrich_coefficients_batched(sample_data_torch, params)
        loop_result = torch_fc.friedrich_coefficients(sample_data_torch, m=3, r=30.0, coeff=0)
        assert torch.allclose(batched_result[0], loop_result, rtol=RTOL, atol=ATOL)

    def test_all_4_coeffs_match_loop(self, sample_data_torch):
        """All 4 coefficients should match loop over non-batched calls."""
        params = [
            {"m": 3, "r": 30.0, "coeff": 0},
            {"m": 3, "r": 30.0, "coeff": 1},
            {"m": 3, "r": 30.0, "coeff": 2},
            {"m": 3, "r": 30.0, "coeff": 3},
        ]
        batched_result = friedrich_coefficients_batched(sample_data_torch, params)

        for idx, p in enumerate(params):
            loop_result = torch_fc.friedrich_coefficients(
                sample_data_torch, m=p["m"], r=p["r"], coeff=p["coeff"]
            )
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at coeff {p['coeff']}"
            )

    def test_empty_params(self, sample_data_torch):
        """Empty params should return empty tensor."""
        result = friedrich_coefficients_batched(sample_data_torch, [])
        assert result.shape == (0, 1, 1)

    def test_batched_data(self, sample_data_torch_batched):
        """Should work with batched data."""
        params = [
            {"m": 3, "r": 30.0, "coeff": 0},
            {"m": 3, "r": 30.0, "coeff": 1},
        ]
        batched_result = friedrich_coefficients_batched(sample_data_torch_batched, params)
        assert batched_result.shape == (2, 4, 3)


# =============================================================================
# PHASE 3: LOW PRIORITY BATCHED FUNCTION TESTS
# =============================================================================


class TestNumberCrossingMBatched:
    """Tests for number_crossing_m_batched function."""

    def test_single_m_matches_loop(self, sample_data_torch):
        """Single m value should match non-batched version."""
        ms = [0.0]
        batched_result = number_crossing_m_batched(sample_data_torch, ms)
        loop_result = torch_fc.number_crossing_m(sample_data_torch, m=0.0)
        assert torch.allclose(batched_result[0], loop_result, rtol=RTOL, atol=ATOL)

    def test_multiple_ms_match_loop(self, sample_data_torch):
        """Multiple m values should match loop over non-batched calls."""
        ms = [-1.0, 0.0, 1.0]
        batched_result = number_crossing_m_batched(sample_data_torch, ms)

        for idx, m in enumerate(ms):
            loop_result = torch_fc.number_crossing_m(sample_data_torch, m=m)
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at m {m}"
            )


class TestC3Batched:
    """Tests for c3_batched function."""

    def test_single_lag_matches_loop(self, sample_data_torch):
        """Single lag should match non-batched version."""
        lags = [1]
        batched_result = c3_batched(sample_data_torch, lags)
        loop_result = torch_fc.c3(sample_data_torch, lag=1)
        assert torch.allclose(batched_result[0], loop_result, rtol=RTOL, atol=ATOL)

    def test_multiple_lags_match_loop(self, sample_data_torch):
        """Multiple lags should match loop over non-batched calls."""
        lags = [1, 2, 3]
        batched_result = c3_batched(sample_data_torch, lags)

        for idx, lag in enumerate(lags):
            loop_result = torch_fc.c3(sample_data_torch, lag=lag)
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at lag {lag}"
            )


class TestTimeReversalAsymmetryStatisticBatched:
    """Tests for time_reversal_asymmetry_statistic_batched function."""

    def test_single_lag_matches_loop(self, sample_data_torch):
        """Single lag should match non-batched version."""
        lags = [1]
        batched_result = time_reversal_asymmetry_statistic_batched(sample_data_torch, lags)
        loop_result = torch_fc.time_reversal_asymmetry_statistic(sample_data_torch, lag=1)
        assert torch.allclose(batched_result[0], loop_result, rtol=RTOL, atol=ATOL)

    def test_multiple_lags_match_loop(self, sample_data_torch):
        """Multiple lags should match loop over non-batched calls."""
        lags = [1, 2, 3]
        batched_result = time_reversal_asymmetry_statistic_batched(sample_data_torch, lags)

        for idx, lag in enumerate(lags):
            loop_result = torch_fc.time_reversal_asymmetry_statistic(sample_data_torch, lag=lag)
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at lag {lag}"
            )


class TestValueCountBatched:
    """Tests for value_count_batched function."""

    def test_single_value_matches_loop(self, sample_data_torch):
        """Single value should match non-batched version."""
        values = [0.0]
        batched_result = value_count_batched(sample_data_torch, values)
        loop_result = torch_fc.value_count(sample_data_torch, value=0.0)
        assert torch.allclose(batched_result[0], loop_result, rtol=RTOL, atol=ATOL)

    def test_multiple_values_match_loop(self, sample_data_torch):
        """Multiple values should match loop over non-batched calls."""
        values = [-1.0, 0.0, 1.0]
        batched_result = value_count_batched(sample_data_torch, values)

        for idx, value in enumerate(values):
            loop_result = torch_fc.value_count(sample_data_torch, value=value)
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at value {value}"
            )


class TestRangeCountBatched:
    """Tests for range_count_batched function."""

    def test_single_range_matches_loop(self, sample_data_torch):
        """Single range should match non-batched version."""
        params = [{"min_val": -1.0, "max_val": 1.0}]
        batched_result = range_count_batched(sample_data_torch, params)
        loop_result = torch_fc.range_count(sample_data_torch, min_val=-1.0, max_val=1.0)
        assert torch.allclose(batched_result[0], loop_result, rtol=RTOL, atol=ATOL)

    def test_multiple_ranges_match_loop(self, sample_data_torch):
        """Multiple ranges should match loop over non-batched calls."""
        params = [
            {"min_val": -1.0, "max_val": 1.0},
            {"min_val": 0.0, "max_val": 2.0},
            {"min_val": -2.0, "max_val": 0.0},
        ]
        batched_result = range_count_batched(sample_data_torch, params)

        for idx, p in enumerate(params):
            loop_result = torch_fc.range_count(
                sample_data_torch, min_val=p["min_val"], max_val=p["max_val"]
            )
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at range {p}"
            )


class TestMeanNAbsoluteMaxBatched:
    """Tests for mean_n_absolute_max_batched function."""

    def test_single_n_matches_loop(self, sample_data_torch):
        """Single n should match non-batched version."""
        ns = [1]
        batched_result = mean_n_absolute_max_batched(sample_data_torch, ns)
        loop_result = torch_fc.mean_n_absolute_max(sample_data_torch, number_of_maxima=1)
        assert torch.allclose(batched_result[0], loop_result, rtol=RTOL, atol=ATOL)

    def test_multiple_ns_match_loop(self, sample_data_torch):
        """Multiple n values should match loop over non-batched calls."""
        ns = [1, 3, 5]
        batched_result = mean_n_absolute_max_batched(sample_data_torch, ns)

        for idx, n in enumerate(ns):
            loop_result = torch_fc.mean_n_absolute_max(sample_data_torch, number_of_maxima=n)
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at n {n}"
            )


class TestAggAutocorrelationBatched:
    """Tests for agg_autocorrelation_batched function."""

    def test_single_param_matches_loop(self, sample_data_torch):
        """Single param should match non-batched version."""
        params = [{"maxlag": 40, "f_agg": "mean"}]
        batched_result = agg_autocorrelation_batched(sample_data_torch, params)
        loop_result = torch_fc.agg_autocorrelation(sample_data_torch, maxlag=40, f_agg="mean")
        assert torch.allclose(batched_result[0], loop_result, rtol=RTOL, atol=ATOL)

    def test_all_aggs_match_loop(self, sample_data_torch):
        """All 3 aggregation types should match loop over non-batched calls."""
        params = [
            {"maxlag": 40, "f_agg": "mean"},
            {"maxlag": 40, "f_agg": "median"},
            {"maxlag": 40, "f_agg": "var"},
        ]
        batched_result = agg_autocorrelation_batched(sample_data_torch, params)

        for idx, p in enumerate(params):
            loop_result = torch_fc.agg_autocorrelation(
                sample_data_torch, maxlag=p["maxlag"], f_agg=p["f_agg"]
            )
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at f_agg {p['f_agg']}"
            )


class TestAugmentedDickeyFullerBatched:
    """Tests for augmented_dickey_fuller_batched function."""

    def test_single_attr_matches_loop(self, sample_data_torch):
        """Single attr should match non-batched version."""
        attrs = ["teststat"]
        batched_result = augmented_dickey_fuller_batched(sample_data_torch, attrs)
        loop_result = torch_fc.augmented_dickey_fuller(sample_data_torch, attr="teststat")
        assert torch.allclose(batched_result[0], loop_result, rtol=RTOL, atol=ATOL)

    def test_all_attrs_match_loop(self, sample_data_torch):
        """All 3 attributes should match loop over non-batched calls."""
        attrs = ["teststat", "pvalue", "usedlag"]
        batched_result = augmented_dickey_fuller_batched(sample_data_torch, attrs)

        for idx, attr in enumerate(attrs):
            loop_result = torch_fc.augmented_dickey_fuller(sample_data_torch, attr=attr)
            assert torch.allclose(batched_result[idx], loop_result, rtol=RTOL, atol=ATOL), (
                f"Mismatch at attr {attr}"
            )
