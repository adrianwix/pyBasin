from abc import ABC, abstractmethod

import numpy as np
import torch


class Sampler(ABC):
    """Abstract base class for sampling initial conditions using PyTorch."""

    def __init__(self, min_limits: list[float], max_limits: list[float]):
        """
        Initialize the sampler.

        :param min_limits: List of minimum values for each state.
        :param max_limits: List of maximum values for each state.
        """
        assert len(min_limits) == len(max_limits), (
            "min_limits and max_limits must have the same length"
        )
        self.min_limits = torch.tensor(min_limits)
        self.max_limits = torch.tensor(max_limits)
        self.state_dim = len(min_limits)

    @abstractmethod
    def sample(self, N: int) -> torch.Tensor:
        """
        Generate N samples for the initial conditions.

        :param N: Number of samples.
        :return: Sampled initial conditions as a tensor of shape (N, state_dim).
        """
        pass

    # def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
    #     """Convert PyTorch tensor to numpy array for compatibility."""
    #     return tensor.numpy()


class UniformRandomSampler(Sampler):
    """Generates random samples using a uniform distribution within the specified range."""

    def sample(self, N: int, seed: int | None = 299792458) -> torch.Tensor:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        return (
            torch.rand(N, self.state_dim, generator=generator) * (self.max_limits - self.min_limits)
            + self.min_limits
        )


class GridSampler(Sampler):
    """Generates evenly spaced samples in a grid pattern within the specified range."""

    def sample(self, N: int) -> torch.Tensor:
        n_per_dim = int(np.floor(N ** (1 / self.state_dim)))

        grid_points = [
            torch.linspace(min_val, max_val, n_per_dim)
            for min_val, max_val in zip(self.min_limits, self.max_limits)
        ]

        grid_matrices = torch.meshgrid(*grid_points, indexing="ij")
        points = torch.stack([grid.t().flatten() for grid in grid_matrices], dim=1)

        print(f"Created grid with {len(points)} points ({n_per_dim} points per dimension)")
        return points


class GaussianSampler(Sampler):
    """Generates samples using a Gaussian distribution around the midpoint."""

    def __init__(self, min_limits: list[float], max_limits: list[float], std_factor: float = 0.2):
        super().__init__(min_limits, max_limits)
        self.std_factor = std_factor

    def sample(self, N: int) -> torch.Tensor:
        mean = (self.min_limits + self.max_limits) / 2
        std = self.std_factor * (self.max_limits - self.min_limits)

        samples = torch.normal(mean.repeat(N, 1), std.repeat(N, 1))

        return torch.clamp(samples, self.min_limits, self.max_limits)
