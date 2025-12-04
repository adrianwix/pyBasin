import pytest
import torch

from pybasin.feature_extractors.jax_feature_calculators import (
    COMPREHENSIVE_FEATURE_NAMES,
    MINIMAL_FEATURE_NAMES,
    get_feature_names,
)
from pybasin.feature_extractors.jax_feature_extractor import JaxFeatureExtractor
from pybasin.solution import Solution


def test_jax_feature_extractor_output_shape():
    """Test that JaxFeatureExtractor returns correct output shape with minimal features."""
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    # Use comprehensive=False for minimal 10 features
    extractor = JaxFeatureExtractor(time_steady=0.0, normalize=False, comprehensive=False)
    features = extractor.extract_features(solution)

    # 10 minimal features per state * 2 states = 20 features
    n_minimal = len(MINIMAL_FEATURE_NAMES)
    assert features.shape == (n_batch, n_minimal * n_states)


def test_jax_feature_extractor_feature_count():
    """Test that extractor has correct feature count after extraction with minimal features."""
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    # Use comprehensive=False for minimal 10 features
    extractor = JaxFeatureExtractor(comprehensive=False)
    extractor.extract_features(solution)

    # 10 minimal features per state * 2 states = 20 features
    n_minimal = len(MINIMAL_FEATURE_NAMES)
    expected = n_minimal * n_states
    assert extractor.n_features == expected
    assert len(extractor.feature_names) == expected


def test_jax_feature_extractor_feature_names():
    """Test that feature names use per-state naming convention with minimal features."""
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    # Use comprehensive=False for minimal 10 features
    extractor = JaxFeatureExtractor(comprehensive=False)
    extractor.extract_features(solution)

    # Should have features for both states with state_X__ prefix
    feature_names = extractor.feature_names
    n_minimal = len(MINIMAL_FEATURE_NAMES)
    assert len(feature_names) == n_minimal * n_states
    assert feature_names[0] == "state_0__sum_values"
    assert feature_names[n_minimal] == "state_1__sum_values"
    assert all("state_0__" in name or "state_1__" in name for name in feature_names)


def test_jax_feature_extractor_time_filtering():
    """Test that time filtering reduces the time dimension."""
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    # Use comprehensive=False for minimal 10 features
    extractor = JaxFeatureExtractor(time_steady=5.0, normalize=False, comprehensive=False)
    features = extractor.extract_features(solution)

    # Should still return minimal features (10 per state * 2 states)
    n_minimal = len(MINIMAL_FEATURE_NAMES)
    assert features.shape == (n_batch, n_minimal * n_states)


def test_jax_feature_extractor_per_state_features():
    """Test that per-state feature configuration works correctly."""
    n_steps, n_batch, n_states = 100, 5, 3
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    # Configure different features for each state
    # Note: when state_to_features is used, states not specified get default features
    extractor = JaxFeatureExtractor(
        state_to_features={
            0: ["maximum", "standard_deviation"],  # 2 features for state 0
            1: ["mean"],  # 1 feature for state 1
            # State 2 uses default (comprehensive features when comprehensive=True)
        },
        normalize=False,
        comprehensive=False,  # Use minimal for state 2
    )
    features = extractor.extract_features(solution)

    # 2 + 1 + 10 (minimal) = 13 features total
    n_minimal = len(MINIMAL_FEATURE_NAMES)
    assert features.shape == (n_batch, 2 + 1 + n_minimal)
    assert extractor.n_features == 2 + 1 + n_minimal


def test_jax_feature_extractor_normalization():
    """Test that normalization produces zero mean and unit variance."""
    n_steps, n_batch, n_states = 100, 50, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    # Use comprehensive=False for faster test
    extractor = JaxFeatureExtractor(normalize=True, comprehensive=False)
    features = extractor.extract_features(solution)

    # After normalization, mean should be ~0 and std ~1
    # Allow wider tolerance for small batch sizes
    assert torch.abs(features.mean()).item() < 0.2
    assert torch.abs(features.std() - 1.0).item() < 0.2


def test_jax_feature_extractor_reset_scaler():
    """Test that reset_scaler resets normalization state."""
    extractor = JaxFeatureExtractor(normalize=True)

    n_steps, n_batch, n_states = 100, 5, 2
    solution = Solution(
        initial_condition=torch.randn(n_batch, n_states),
        time=torch.linspace(0, 10, n_steps),
        y=torch.randn(n_steps, n_batch, n_states),
    )

    extractor.extract_features(solution)
    assert extractor._is_fitted  # pyright: ignore[reportPrivateUsage]

    extractor.reset_scaler()
    assert not extractor._is_fitted  # pyright: ignore[reportPrivateUsage]
    assert extractor._feature_mean is None  # pyright: ignore[reportPrivateUsage]
    assert extractor._feature_std is None  # pyright: ignore[reportPrivateUsage]


def test_jax_feature_extractor_comprehensive_features():
    """Test that comprehensive feature set includes all features per state."""
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    # Use comprehensive features (includes delta and log_delta)
    comprehensive_names = get_feature_names(comprehensive=True)
    n_comprehensive = len(COMPREHENSIVE_FEATURE_NAMES)
    extractor = JaxFeatureExtractor(
        default_features=comprehensive_names,
        normalize=False,
    )
    features = extractor.extract_features(solution)

    # n_comprehensive features per state * 2 states
    expected_features = n_comprehensive * n_states
    assert features.shape == (n_batch, expected_features)
    assert extractor.n_features == expected_features

    # Verify delta and log_delta are included
    feature_names = extractor.feature_names
    assert "state_0__delta" in feature_names
    assert "state_0__log_delta" in feature_names
    assert "state_1__delta" in feature_names
    assert "state_1__log_delta" in feature_names


def test_jax_feature_extractor_custom_feature_in_per_state():
    """Test that custom features (delta, log_delta) work in per-state configuration."""
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    # Use log_delta for state 1 (like pendulum case)
    extractor = JaxFeatureExtractor(
        state_to_features={
            0: [],  # No features for state 0
            1: ["log_delta"],  # Only log_delta for state 1
        },
        normalize=False,
    )
    features = extractor.extract_features(solution)

    # Only 1 feature total
    assert features.shape == (n_batch, 1)
    assert extractor.n_features == 1
    assert extractor.feature_names == ["state_1__log_delta"]


def test_jax_feature_extractor_feature_names_format():
    """Test that feature_names property returns correctly formatted names."""
    n_steps, n_batch, n_states = 100, 5, 3
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = JaxFeatureExtractor(
        state_to_features={
            0: ["maximum", "minimum"],
            1: ["mean", "variance", "standard_deviation"],
            2: ["delta"],
        },
        normalize=False,
    )
    extractor.extract_features(solution)

    expected_names = [
        "state_0__maximum",
        "state_0__minimum",
        "state_1__mean",
        "state_1__variance",
        "state_1__standard_deviation",
        "state_2__delta",
    ]
    assert extractor.feature_names == expected_names
    assert extractor.n_features == 6


def test_jax_feature_extractor_feature_names_raises_before_extraction():
    """Test that accessing feature_names before extraction raises RuntimeError."""
    extractor = JaxFeatureExtractor()

    with pytest.raises(RuntimeError, match="Feature configuration not initialized"):
        _ = extractor.feature_names

    with pytest.raises(RuntimeError, match="Feature configuration not initialized"):
        _ = extractor.n_features
