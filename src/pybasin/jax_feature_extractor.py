"""JAX-based feature extractor for ODE solution trajectories.

This module provides a high-performance feature extractor using JAX for GPU-accelerated
time series feature extraction from ODE solutions, matching tsfresh's MinimalFCParameters.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import Array

from pybasin.feature_extractor import FeatureExtractor
from pybasin.jax_feature_calculators import MINIMAL_FEATURES, get_feature_names
from pybasin.jax_utils import get_jax_device, jax_to_torch, torch_to_jax
from pybasin.solution import Solution


class JaxFeatureExtractor(FeatureExtractor):
    """JAX-based feature extractor using MinimalFCParameters features.

    Args:
        time_steady: Time threshold for filtering transients. Default 0.0.
        exclude_states: Optional list of state indices to exclude.
        normalize: Whether to apply z-score normalization. Default True.
        use_jit: Whether to JIT-compile extraction. Default True.
        device: JAX device to use ('cpu', 'gpu', 'cuda', 'cuda:N', or None for auto).
    """

    def __init__(
        self,
        time_steady: float = 0.0,
        exclude_states: list[int] | None = None,
        normalize: bool = True,
        use_jit: bool = True,
        device: str | None = None,
    ):
        import threading

        super().__init__(time_steady=time_steady, exclude_states=exclude_states)

        self.normalize = normalize
        self.use_jit = use_jit
        self._is_fitted = False
        self._fit_lock = threading.Lock()

        self._feature_mean: Array | None = None
        self._feature_std: Array | None = None

        self.features: dict[str, Callable[[Array], Array]] = MINIMAL_FEATURES.copy()
        self.feature_names = get_feature_names()

        # Resolve JAX device
        self.jax_device = get_jax_device(device)

        self._extract_fn: Callable[[Array], Array] = self._build_extract_function()

    def _build_extract_function(self) -> Callable[[Array], Array]:
        """Build the feature extraction function."""
        feature_funcs = list(self.features.values())

        def extract_all_features(x: Array) -> Array:
            feature_values = [func(x) for func in feature_funcs]
            stacked = jnp.stack(feature_values, axis=0)  # (num_features, B, S)
            transposed = jnp.transpose(stacked, (1, 0, 2))  # (B, num_features, S)
            b = transposed.shape[0]
            return transposed.reshape(b, -1)  # (B, num_features * S)

        if self.use_jit:
            return jax.jit(extract_all_features)  # type: ignore[return-value]
        return extract_all_features

    def extract_features(self, solution: Solution) -> torch.Tensor:
        """Extract features from an ODE solution using JAX."""
        # Apply time filtering using numpy (before JAX conversion)
        y = solution.y
        time_np = solution.time.cpu().numpy()

        if self.time_steady > 0:
            idx_steady = int(np.searchsorted(time_np, self.time_steady, side="right"))
            y = y[idx_steady:]

        # Apply state filtering
        if self.exclude_states is not None:
            keep_indices = [i for i in range(y.shape[2]) if i not in self.exclude_states]
            y = y[..., keep_indices]

        # Convert to JAX (uses DLPack for zero-copy on GPU)
        y_jax = torch_to_jax(y, self.jax_device)
        features_jax: Array = self._extract_fn(y_jax)

        # Apply normalization
        if self.normalize:
            with self._fit_lock:
                if not self._is_fitted:
                    self._feature_mean = jnp.mean(features_jax, axis=0, keepdims=True)
                    self._feature_std = jnp.std(features_jax, axis=0, keepdims=True)
                    self._feature_std = jnp.where(self._feature_std == 0, 1.0, self._feature_std)
                    self._is_fitted = True

                assert self._feature_mean is not None
                assert self._feature_std is not None
                features_jax = (features_jax - self._feature_mean) / self._feature_std

        # Convert back to PyTorch
        return jax_to_torch(features_jax)

    def reset_scaler(self) -> None:
        """Reset the normalization parameters."""
        with self._fit_lock:
            self._feature_mean = None
            self._feature_std = None
            self._is_fitted = False

    @property
    def n_features(self) -> int:
        """Return the number of features per state variable."""
        return len(self.features)
