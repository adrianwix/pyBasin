import torch

from pybasin.feature_extractors.jax_feature_extractor import JaxFeatureExtractor
from pybasin.solution import Solution


def test_jax_feature_extractor_output_shape():
    """Test that JaxFeatureExtractor returns correct output shape."""
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = JaxFeatureExtractor(time_steady=0.0, normalize=False)
    features = extractor.extract_features(solution)

    # 10 features per state * 2 states = 20 features
    assert features.shape == (n_batch, 20)


def test_jax_feature_extractor_feature_count():
    """Test that extractor has correct feature count after extraction."""
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = JaxFeatureExtractor()
    extractor.extract_features(solution)

    # 10 features per state * 2 states = 20 features
    assert extractor.n_features == 20
    assert len(extractor.feature_names) == 20


def test_jax_feature_extractor_feature_names():
    """Test that feature names use per-state naming convention."""
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = JaxFeatureExtractor()
    extractor.extract_features(solution)

    # Should have features for both states with state_X__ prefix
    feature_names = extractor.feature_names
    assert len(feature_names) == 20  # 10 features * 2 states
    assert feature_names[0] == "state_0__sum_values"
    assert feature_names[10] == "state_1__sum_values"
    assert all("state_0__" in name or "state_1__" in name for name in feature_names)


def test_jax_feature_extractor_time_filtering():
    """Test that time filtering reduces the time dimension."""
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = JaxFeatureExtractor(time_steady=5.0, normalize=False)
    features = extractor.extract_features(solution)

    # Should still return 20 features (10 per state * 2 states)
    assert features.shape == (n_batch, 20)


def test_jax_feature_extractor_per_state_features():
    """Test that per-state feature configuration works correctly."""
    n_steps, n_batch, n_states = 100, 5, 3
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    # Configure different features for each state
    extractor = JaxFeatureExtractor(
        state_to_features={
            0: ["maximum", "standard_deviation"],  # 2 features for state 0
            1: ["mean"],  # 1 feature for state 1
            # State 2 uses default (all 10 features)
        },
        normalize=False,
    )
    features = extractor.extract_features(solution)

    # 2 + 1 + 10 = 13 features total
    assert features.shape == (n_batch, 13)
    assert extractor.n_features == 13


def test_jax_feature_extractor_normalization():
    """Test that normalization produces zero mean and unit variance."""
    n_steps, n_batch, n_states = 100, 50, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = JaxFeatureExtractor(normalize=True)
    features = extractor.extract_features(solution)

    # After normalization, mean should be ~0 and std ~1
    assert torch.abs(features.mean()).item() < 0.1
    assert torch.abs(features.std() - 1.0).item() < 0.1


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
