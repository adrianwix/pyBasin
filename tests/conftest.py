"""Test configuration and fixtures for pybasin tests."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def artifacts_dir(project_root):
    """Get the artifacts directory."""
    return project_root / "artifacts"


@pytest.fixture
def matlab_results_dir(project_root):
    """Get the directory containing MATLAB comparison results."""
    matlab_dir = project_root.parent.parent / "bSTAB-M"
    if matlab_dir.exists():
        return matlab_dir
    return None


@pytest.fixture
def tolerance():
    """Default tolerance for comparing basin stability values."""
    return 0.05  # 5% tolerance


@pytest.fixture
def random_seed():
    """Fixed random seed for reproducibility."""
    seed = 42
    np.random.seed(seed)
    return seed


@pytest.fixture
def sample_initial_conditions():
    """Generate sample initial conditions for testing."""

    def _generator(bounds, n_samples=100, seed=42):
        """Generate random initial conditions within bounds.

        Args:
            bounds: List of (min, max) tuples for each dimension
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        dim = len(bounds)
        samples = np.zeros((n_samples, dim))
        for i, (min_val, max_val) in enumerate(bounds):
            samples[:, i] = np.random.uniform(min_val, max_val, n_samples)
        return samples

    return _generator
