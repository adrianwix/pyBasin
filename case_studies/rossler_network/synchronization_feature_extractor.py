"""Synchronization feature extractor for network systems.

This module provides a feature extractor that computes synchronization metrics
from trajectories of coupled oscillator networks.
"""

import jax.numpy as jnp
import torch
from jax import Array

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.jax_utils import get_jax_device, jax_to_torch, torch_to_jax
from pybasin.solution import Solution


class SynchronizationFeatureExtractor(FeatureExtractor):
    """
    Feature extractor that computes synchronization metrics for network systems.

    For a network of N oscillators with 3 states each (x, y, z), computes:
    - max_deviation_x: max_i,j |x_i - x_j| over steady-state
    - max_deviation_y: max_i,j |y_i - y_j| over steady-state
    - max_deviation_z: max_i,j |z_i - z_j| over steady-state
    - max_deviation_all: maximum of the above three

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the network.
    time_steady : float
        Time after which the transient is considered over.
    device : str, optional
        Device for computation ('cpu', 'cuda', etc.).
    """

    def __init__(
        self,
        n_nodes: int,
        time_steady: float = 1000.0,
        device: str | None = None,
    ):
        super().__init__(time_steady=time_steady)
        self.n_nodes = n_nodes
        self.jax_device = get_jax_device(device)
        self._feature_names = [
            "max_deviation_x",
            "max_deviation_y",
            "max_deviation_z",
            "max_deviation_all",
        ]

    @property
    def feature_names(self) -> list[str]:
        """Return the names of the extracted features."""
        return self._feature_names

    def extract_features(self, solution: Solution) -> torch.Tensor:
        """
        Extract synchronization features from trajectories.

        Parameters
        ----------
        solution : Solution
            Solution containing trajectory data with shape (n_times, n_samples, 3*N)

        Returns
        -------
        torch.Tensor
            Feature matrix with shape (n_samples, 4)
        """
        y_filtered = self.filter_time(solution)

        y_jax = torch_to_jax(y_filtered, self.jax_device)

        y_transposed = jnp.transpose(y_jax, (1, 0, 2))

        features_jax = self._compute_sync_features(y_transposed)

        features_torch = jax_to_torch(features_jax)
        solution.extracted_features = features_torch
        solution.extracted_feature_names = self._feature_names
        solution.features = features_torch
        solution.filtered_feature_names = self._feature_names

        return features_torch

    def _compute_sync_features(self, y_steady: Array) -> Array:
        """
        Compute synchronization features for all trajectories.

        Uses only the FINAL time step to measure synchronization state,
        avoiding penalization of trajectories still converging during the window.

        Parameters
        ----------
        y_steady : Array
            Steady-state trajectories with shape (n_samples, n_steady_times, 3*N)

        Returns
        -------
        Array
            Features with shape (n_samples, 4)
        """
        N = self.n_nodes

        y_final = y_steady[:, -1, :]

        x = y_final[:, :N]
        y = y_final[:, N : 2 * N]
        z = y_final[:, 2 * N :]

        max_dev_x = jnp.max(x, axis=1) - jnp.min(x, axis=1)
        max_dev_y = jnp.max(y, axis=1) - jnp.min(y, axis=1)
        max_dev_z = jnp.max(z, axis=1) - jnp.min(z, axis=1)

        max_dev_all = jnp.maximum(jnp.maximum(max_dev_x, max_dev_y), max_dev_z)

        features = jnp.stack([max_dev_x, max_dev_y, max_dev_z, max_dev_all], axis=1)

        return features
