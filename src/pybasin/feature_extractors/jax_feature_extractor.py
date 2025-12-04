"""JAX-based feature extractor for ODE solution trajectories.

This module provides a high-performance feature extractor using JAX for GPU-accelerated
time series feature extraction from ODE solutions.
"""

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import Array

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.feature_extractors.jax_feature_calculators import ALL_FEATURES, get_feature_names
from pybasin.feature_extractors.jax_feature_utilities import impute, impute_extreme
from pybasin.jax_utils import get_jax_device, jax_to_torch, torch_to_jax
from pybasin.solution import Solution


class JaxFeatureExtractor(FeatureExtractor):
    """JAX-based feature extractor for time series features.

    Supports per-state variable feature configuration, allowing you to apply
    different feature sets to different state variables based on domain knowledge.

    By default, uses COMPREHENSIVE_FEATURES which includes all tsfresh EfficientFCParameters
    equivalent features plus custom features (delta, log_delta).

    Args:
        time_steady: Time threshold for filtering transients. Default 0.0.
        comprehensive: If True (default), use all COMPREHENSIVE_FEATURES (tsfresh
            EfficientFCParameters equivalent + custom features). If False, use only
            MINIMAL_FEATURES (tsfresh MinimalFCParameters equivalent).
        default_features: Default feature calculators to apply to all states.
            Can be a list of feature names or None. If None, features are determined
            by the `comprehensive` parameter.
            Example: ["maximum", "standard_deviation"]
        state_to_features: Optional dict mapping state indices to feature lists.
            Allows different feature sets per state variable. If provided, overrides
            default_features for those states. Example:
            {
                0: ["maximum", "standard_deviation"],  # State 0: only max and std
                1: ["mean", "median", "variance"],     # State 1: different features
            }
            States not in this dict will use default_features.
        normalize: Whether to apply z-score normalization. Default True.
        use_jit: Whether to JIT-compile extraction. Default True.
        device: JAX device to use ('cpu', 'gpu', 'cuda', 'cuda:N', or None for auto).
        impute_method: Method for handling NaN/inf values in features. Options:
            - 'extreme': Replace with extreme values (1e10) to distinguish unbounded
              trajectories. Best for systems with divergent solutions. (default)
            - 'tsfresh': Replace using tsfresh-style imputation (inf->max/min,
              NaN->median). Better when all trajectories are bounded.

    Examples:
        >>> # Use all comprehensive features (default) for all states
        >>> extractor = JaxFeatureExtractor(time_steady=9.0)

        >>> # Use only minimal features
        >>> extractor = JaxFeatureExtractor(time_steady=9.0, comprehensive=False)

        >>> # Custom features for specific states
        >>> extractor = JaxFeatureExtractor(
        ...     time_steady=9.0,
        ...     state_to_features={
        ...         0: ["maximum", "standard_deviation"],  # Position
        ...         1: ["mean"],                           # Velocity
        ...     },
        ... )
    """

    def __init__(
        self,
        time_steady: float = 0.0,
        comprehensive: bool = True,
        default_features: list[str] | None = None,
        state_to_features: dict[int, list[str]] | None = None,
        normalize: bool = True,
        use_jit: bool = True,
        device: str | None = None,
        impute_method: Literal["extreme", "tsfresh"] = "extreme",
    ):
        import threading

        super().__init__(time_steady=time_steady)

        self.comprehensive = comprehensive
        self.normalize = normalize
        self.use_jit = use_jit
        self.impute_method = impute_method
        self._is_fitted = False
        self._fit_lock = threading.Lock()

        self._feature_mean: Array | None = None
        self._feature_std: Array | None = None

        # Configure features per state
        self.default_features = default_features
        self.state_to_features = state_to_features or {}

        # Build the feature configuration
        # This will be finalized when we know the number of states (in extract_features)
        self._state_feature_config: dict[int, list[str]] | None = None
        self._num_states: int | None = None

        # Resolve JAX device
        self.jax_device = get_jax_device(device)

        # Extract function will be built on first call
        self._extract_fn: Callable[[Array], Array] | None = None

    def _configure_state_features(self, num_states: int) -> None:
        """Configure which features to compute for each state."""
        if self._state_feature_config is not None and self._num_states == num_states:
            return  # Already configured

        self._num_states = num_states
        self._state_feature_config = {}

        # Determine default features based on comprehensive flag or explicit list
        if self.default_features is None:
            default_feature_list = get_feature_names(comprehensive=self.comprehensive)
        else:
            default_feature_list = self.default_features

        # Configure each state
        for state_idx in range(num_states):
            if state_idx in self.state_to_features:
                self._state_feature_config[state_idx] = self.state_to_features[state_idx]
            else:
                self._state_feature_config[state_idx] = default_feature_list

        # Build the extraction function with the configured features
        self._extract_fn = self._build_extract_function()

    def _build_extract_function(self) -> Callable[[Array], Array]:
        """Build the feature extraction function based on per-state configuration."""
        if self._state_feature_config is None or self._num_states is None:
            raise RuntimeError(
                "Must call _configure_state_features before building extract function"
            )

        # Build a list of (state_idx, feature_func) tuples
        state_feature_funcs: list[tuple[int, Callable[[Array], Array]]] = []
        for state_idx in range(self._num_states):
            feature_names = self._state_feature_config[state_idx]
            for fname in feature_names:
                if fname not in ALL_FEATURES:
                    available = list(ALL_FEATURES.keys())
                    raise ValueError(f"Unknown feature: {fname}. Available: {available}")
                state_feature_funcs.append((state_idx, ALL_FEATURES[fname]))

        def extract_all_features(x: Array) -> Array:
            # x shape: (N, B, S) where S is number of states
            # For each (state_idx, feature_func), extract feature for that state
            feature_values: list[Array] = []
            for state_idx, feature_func in state_feature_funcs:
                # Extract single state keeping dim: (N, B, 1)
                x_state = x[:, :, state_idx : state_idx + 1]
                # Compute feature: (B, 1)
                feat = feature_func(x_state)
                # Squeeze state dim: (B,)
                feature_values.append(feat.squeeze(-1))

            # Stack features: (num_features, B)
            stacked = jnp.stack(feature_values, axis=0)
            # Transpose: (B, num_features)
            return jnp.transpose(stacked, (1, 0))

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

        # Configure per-state features on first call
        num_states = y.shape[2]
        if self._extract_fn is None:
            self._configure_state_features(num_states)

        assert self._extract_fn is not None

        # Convert to JAX (uses DLPack for zero-copy on GPU)
        y_jax = torch_to_jax(y, self.jax_device)
        features_jax: Array = self._extract_fn(y_jax)

        # Handle NaN and inf values using the configured impute method
        if self.impute_method == "extreme":
            # Replace with extreme values so unbounded trajectories are distinguishable
            features_jax = impute_extreme(features_jax)
        else:
            # tsfresh-style: inf->max/min, NaN->median
            features_jax = impute(features_jax)

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
        """Return the total number of features across all states."""
        if self._state_feature_config is None or self._num_states is None:
            raise RuntimeError(
                "Feature configuration not initialized. Call extract_features first."
            )

        total = 0
        for state_idx in range(self._num_states):
            total += len(self._state_feature_config[state_idx])
        return total

    @property
    def feature_names(self) -> list[str]:
        """Return the list of feature names in the format 'state_X__feature_name'."""
        if self._state_feature_config is None or self._num_states is None:
            raise RuntimeError(
                "Feature configuration not initialized. Call extract_features first."
            )

        names: list[str] = []
        for state_idx in range(self._num_states):
            for fname in self._state_feature_config[state_idx]:
                names.append(f"state_{state_idx}__{fname}")
        return names
