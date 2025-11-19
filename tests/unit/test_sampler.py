from typing import TypedDict

import pytest
import torch

from pybasin.sampler import GaussianSampler, GridSampler, UniformRandomSampler


class SamplerParams(TypedDict):
    min_limits: list[float]
    max_limits: list[float]
    device: str


@pytest.fixture
def sampler_params() -> SamplerParams:
    return {
        "min_limits": [0.0, -1.0],
        "max_limits": [1.0, 1.0],
        "device": "cpu",
    }


def test_uniform_sampler_shape(sampler_params: SamplerParams) -> None:
    sampler = UniformRandomSampler(**sampler_params)
    n = 100
    samples = sampler.sample(n)

    # Verify samples has correct shape: 100 samples × 2 dimensions
    assert samples.shape == (n, 2)
    # Confirm samples are on CPU device as requested
    assert samples.device.type == "cpu"
    # Validate data type is float32 for efficiency
    assert samples.dtype == torch.float32


def test_uniform_sampler_bounds(sampler_params: SamplerParams) -> None:
    sampler = UniformRandomSampler(**sampler_params)
    samples = sampler.sample(1000)

    # First dimension samples are within [0.0, 1.0] bounds
    assert torch.all(samples[:, 0] >= 0.0) and torch.all(samples[:, 0] <= 1.0)
    # Second dimension samples are within [-1.0, 1.0] bounds
    assert torch.all(samples[:, 1] >= -1.0) and torch.all(samples[:, 1] <= 1.0)


def test_uniform_sampler_seed_reproducibility(sampler_params: SamplerParams) -> None:
    sampler = UniformRandomSampler(**sampler_params)
    samples1 = sampler.sample(100, seed=42)
    samples2 = sampler.sample(100, seed=42)

    # Same seed produces identical samples (reproducibility)
    assert torch.allclose(samples1, samples2)


def test_grid_sampler_coverage(sampler_params: SamplerParams) -> None:
    sampler = GridSampler(**sampler_params)
    samples = sampler.sample(100)

    # Grid has correct number of dimensions (2)
    assert samples.shape[1] == 2
    # Grid has at least 100 points (may have more due to ceiling operation)
    assert samples.shape[0] >= 100


def test_gaussian_sampler_distribution(sampler_params: SamplerParams) -> None:
    sampler = GaussianSampler(**sampler_params, std_factor=0.2)
    samples = sampler.sample(1000)

    mean = samples.mean(dim=0)
    expected_mean = torch.tensor([0.5, 0.0])
    # Sample mean ≈ [0.5, 0.0] (midpoint between bounds)
    assert torch.allclose(mean, expected_mean, atol=0.1)
