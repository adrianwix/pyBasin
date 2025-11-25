import torch

from pybasin.jax_feature_extractor import JaxFeatureExtractor
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
    """Test that extractor has 10 minimal features."""
    extractor = JaxFeatureExtractor()

    assert extractor.n_features == 10
    assert len(extractor.feature_names) == 10


def test_jax_feature_extractor_feature_names():
    """Test that feature names match tsfresh MinimalFCParameters."""
    extractor = JaxFeatureExtractor()

    expected_names = [
        "sum_values",
        "median",
        "mean",
        "length",
        "standard_deviation",
        "variance",
        "root_mean_square",
        "maximum",
        "absolute_maximum",
        "minimum",
    ]
    assert extractor.feature_names == expected_names


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


def test_jax_feature_extractor_state_filtering():
    """Test that state filtering reduces the state dimension."""
    n_steps, n_batch, n_states = 100, 5, 3
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = JaxFeatureExtractor(exclude_states=[2], normalize=False)
    features = extractor.extract_features(solution)

    # 10 features per state * 2 remaining states = 20 features
    assert features.shape == (n_batch, 20)


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
