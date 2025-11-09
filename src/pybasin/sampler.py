from abc import ABC, abstractmethod

import numpy as np
import torch


class Sampler(ABC):
    """Abstract base class for sampling initial conditions using PyTorch."""

    def __init__(self, min_limits: list[float], max_limits: list[float], device: str | None = None):
        """
        Initialize the sampler.

        :param min_limits: List of minimum values for each state.
        :param max_limits: List of maximum values for each state.
        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        assert len(min_limits) == len(max_limits), (
            "min_limits and max_limits must have the same length"
        )

        # Auto-detect device if not specified and normalize cuda to cuda:0
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            # Normalize "cuda" to "cuda:0" for consistency
            dev = torch.device(device)
            if dev.type == "cuda" and dev.index is None:
                self.device = torch.device("cuda:0")
            else:
                self.device = dev

        # Use float32 for GPU efficiency (5-10x faster than float64)
        self.min_limits = torch.tensor(min_limits, dtype=torch.float32, device=self.device)
        self.max_limits = torch.tensor(max_limits, dtype=torch.float32, device=self.device)
        self.state_dim = len(min_limits)

    @abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """
        Generate n samples for the initial conditions.

        :param n: Number of samples.
        :return: Sampled initial conditions as a tensor of shape (n, state_dim).
        """
        pass


class UniformRandomSampler(Sampler):
    """Generates random samples using a uniform distribution within the specified range."""

    def sample(self, n: int, seed: int | None = 299792458) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        return (
            torch.rand(n, self.state_dim, generator=generator, device=self.device)
            * (self.max_limits - self.min_limits)
            + self.min_limits
        )


class GridSampler(Sampler):
    """Generates evenly spaced samples in a grid pattern within the specified range."""

    def sample(self, n: int) -> torch.Tensor:
        # Use ceiling to match MATLAB implementation
        # This ensures we get at least N points (actually N_per_dim^state_dim points)
        n_per_dim = int(np.ceil(n ** (1 / self.state_dim)))

        grid_points = [
            torch.linspace(min_val.item(), max_val.item(), n_per_dim, device=self.device)
            for min_val, max_val in zip(self.min_limits, self.max_limits, strict=True)
        ]

        grid_matrices = torch.meshgrid(*grid_points, indexing="ij")
        points = torch.stack([grid.t().flatten() for grid in grid_matrices], dim=1)

        return points


class GaussianSampler(Sampler):
    """Generates samples using a Gaussian distribution around the midpoint."""

    def __init__(
        self,
        min_limits: list[float],
        max_limits: list[float],
        std_factor: float = 0.2,
        device: str | None = None,
    ):
        super().__init__(min_limits, max_limits, device)
        self.std_factor = std_factor

    def sample(self, n: int) -> torch.Tensor:
        mean = (self.min_limits + self.max_limits) / 2
        std = self.std_factor * (self.max_limits - self.min_limits)

        samples = torch.normal(mean.repeat(n, 1), std.repeat(n, 1))

        return torch.clamp(samples, self.min_limits, self.max_limits)
