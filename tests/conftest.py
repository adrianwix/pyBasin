"""Test configuration and fixtures for pybasin tests."""

import pytest


@pytest.fixture
def tolerance() -> float:
    """Default tolerance for comparing basin stability values."""
    return 0.015  # 1.5% tolerance (increased to account for statistical variability)
