from abc import ABC, abstractmethod
import numpy as np
from typing import List

class Sampler(ABC):
    """Abstract base class for sampling initial conditions."""

    def __init__(self, min_limits: List[float], max_limits: List[float]):
        """
        Initialize the sampler.

        :param min_limits: List of minimum values for each state.
        :param max_limits: List of maximum values for each state.
        """
        assert len(min_limits) == len(max_limits), "min_limits and max_limits must have the same length"
        self.min_limits = np.array(min_limits)
        self.max_limits = np.array(max_limits)
        self.state_dim = len(min_limits)  # Infer the number of states

    @abstractmethod
    def sample(self, N: int) -> np.ndarray:
        """
        Generate N samples for the initial conditions.

        :param N: Number of samples.
        :return: Sampled initial conditions as an array of shape (N, state_dim).
        """
        pass


class RandomSampler(Sampler):
    """Generates random samples using a uniform distribution within the specified range."""

    def sample(self, N: int) -> np.ndarray:
        rng = np.random.default_rng()
        return rng.uniform(self.min_limits, self.max_limits, (N, self.state_dim))


class GridSampler(Sampler):
    """Generates evenly spaced samples in a grid pattern within the specified range."""

    def sample(self, N: int) -> np.ndarray:
        """
        Generate a grid of N^(1/d) points per dimension, where d is the number of dimensions.
        
        :param N: Approximate total number of points desired
        :return: Array of shape (N_actual, state_dim) containing grid points
        """
        # Calculate number of points per dimension
        n_per_dim = int(np.floor(N**(1/self.state_dim)))
        
        # Create grid points for each dimension
        grid_points = [np.linspace(min_val, max_val, n_per_dim) 
                      for min_val, max_val in zip(self.min_limits, self.max_limits)]
        
        # Create meshgrid
        grid_matrices = np.meshgrid(*grid_points)
        
        # Reshape to get all combinations
        points = np.vstack([grid.flatten() for grid in grid_matrices]).T
        
        print(f"Created grid with {len(points)} points ({n_per_dim} points per dimension)")
        
        return points


class GaussianSampler(Sampler):
    """Generates samples using a Gaussian (normal) distribution around the midpoint."""

    def __init__(self, min_limits: List[float], max_limits: List[float], std_factor: float = 0.2):
        """
        :param std_factor: Standard deviation as a fraction of the range.
        """
        super().__init__(min_limits, max_limits)
        self.std_factor = std_factor

    def sample(self, N: int) -> np.ndarray:
        rng = np.random.default_rng()
        mean = (self.min_limits + self.max_limits) / 2
        std = self.std_factor * (self.max_limits - self.min_limits)
        samples = rng.normal(mean, std, (N, self.state_dim))
        return np.clip(samples, self.min_limits, self.max_limits)  # Ensure values stay within limits
