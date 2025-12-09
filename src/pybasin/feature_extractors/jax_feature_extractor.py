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
from pybasin.feature_extractors.jax_feature_calculators import (
    ALL_FEATURE_FUNCTIONS,
    JAX_MINIMAL_FC_PARAMETERS,
    FCParameters,
    get_feature_names_from_config,
)
from pybasin.feature_extractors.jax_feature_utilities import impute, impute_extreme
from pybasin.jax_utils import get_jax_device, jax_to_torch, torch_to_jax
from pybasin.solution import Solution


class JaxFeatureExtractor(FeatureExtractor):
    """JAX-based feature extractor for time series features.

    Supports per-state variable feature configuration using tsfresh-style FCParameters
    dictionaries, allowing different feature sets for different state variables.

    Warning:
        Using JAX_COMPREHENSIVE_FC_PARAMETERS may cause very long JIT compile times
        (~40 minutes). Use JAX_MINIMAL_FC_PARAMETERS or a custom subset for faster
        compilation.

    Args:
        time_steady: Time threshold for filtering transients. Default 0.0.
        features: Default FCParameters configuration to apply to all states.
            Defaults to JAX_MINIMAL_FC_PARAMETERS. Set to None to skip states
            not explicitly configured in features_per_state.
        features_per_state: Optional dict mapping state indices to FCParameters.
            Overrides `features` for specified states. Use None as value to skip
            a state. States not in this dict use the global `features` config.
        normalize: Whether to apply z-score normalization. Default True.
        use_jit: Whether to JIT-compile extraction. Default True.
        device: JAX device to use ('cpu', 'gpu', 'cuda', 'cuda:N', or None for auto).
        impute_method: Method for handling NaN/inf values in features. Options:
            - 'extreme': Replace with extreme values (1e10) to distinguish unbounded
              trajectories. Best for systems with divergent solutions. (default)
            - 'tsfresh': Replace using tsfresh-style imputation (inf->max/min,
              NaN->median). Better when all trajectories are bounded.

    Examples:
        >>> # Default: use minimal features for all states
        >>> extractor = JaxFeatureExtractor(time_steady=9.0)

        >>> # Custom features for specific states, skip others
        >>> extractor = JaxFeatureExtractor(
        ...     time_steady=9.0,
        ...     features=None,  # Don't extract features by default
        ...     features_per_state={
        ...         1: {"log_delta": None},  # Only extract for state 1
        ...     },
        ... )

        >>> # Global features with per-state override
        >>> extractor = JaxFeatureExtractor(
        ...     time_steady=9.0,
        ...     features_per_state={
        ...         0: {"maximum": None},  # Override state 0
        ...         1: None,  # Skip state 1
        ...     },
        ... )
    """

    def __init__(
        self,
        time_steady: float = 0.0,
        features: FCParameters | None = JAX_MINIMAL_FC_PARAMETERS,
        features_per_state: dict[int, FCParameters | None] | None = None,
        normalize: bool = True,
        use_jit: bool = True,
        device: str | None = None,
        impute_method: Literal["extreme", "tsfresh"] = "extreme",
    ):
        import threading

        super().__init__(time_steady=time_steady)

        self.normalize = normalize
        self.use_jit = use_jit
        self.impute_method = impute_method
        self._is_fitted = False
        self._fit_lock = threading.Lock()

        self._feature_mean: Array | None = None
        self._feature_std: Array | None = None

        self.features = features
        self.features_per_state = features_per_state or {}

        self._state_feature_config: dict[int, FCParameters] | None = None
        self._num_states: int | None = None

        self.jax_device = get_jax_device(device)

        self._extract_fn: Callable[[Array], Array] | None = None

    def _configure_state_features(self, num_states: int) -> None:
        """Configure which features to compute for each state."""
        if self._state_feature_config is not None and self._num_states == num_states:
            return

        self._num_states = num_states
        self._state_feature_config = {}

        for state_idx in range(num_states):
            if state_idx in self.features_per_state:
                fc_params = self.features_per_state[state_idx]
                if fc_params is not None:
                    self._state_feature_config[state_idx] = fc_params
            elif self.features is not None:
                self._state_feature_config[state_idx] = self.features

        self._extract_fn = self._build_extract_function()

    def _is_uniform_config(self) -> bool:
        """Check if all states use the same feature configuration."""
        if not self._state_feature_config:
            return True
        configs = list(self._state_feature_config.values())
        if len(configs) <= 1:
            return True
        first = configs[0]
        return all(c is first for c in configs[1:])

    def _build_extract_function(self) -> Callable[[Array], Array]:
        """Build the feature extraction function based on per-state FCParameters."""
        if self._state_feature_config is None or self._num_states is None:
            raise RuntimeError(
                "Must call _configure_state_features before building extract function"
            )

        if not self._state_feature_config:

            def extract_no_features(x: Array) -> Array:
                batch_size = x.shape[1]
                return jnp.zeros((batch_size, 0), dtype=jnp.float32)  # type: ignore[misc]

            if self.use_jit:
                return jax.jit(extract_no_features)  # type: ignore[misc]
            return extract_no_features

        if self._is_uniform_config():
            fc_params = next(iter(self._state_feature_config.values()))
            state_indices = list(self._state_feature_config.keys())

            feature_funcs: list[tuple[Callable[..., Array], dict[str, object] | None]] = []
            for feature_name, param_list in fc_params.items():
                if feature_name not in ALL_FEATURE_FUNCTIONS:
                    available = list(ALL_FEATURE_FUNCTIONS.keys())
                    raise ValueError(f"Unknown feature: {feature_name}. Available: {available}")

                func = ALL_FEATURE_FUNCTIONS[feature_name]
                if param_list is None:
                    feature_funcs.append((func, None))
                else:
                    for params in param_list:
                        feature_funcs.append((func, params))

            def extract_uniform(x: Array) -> Array:
                x_selected = x[:, :, jnp.array(state_indices)]  # type: ignore[misc]
                feature_values: list[Array] = []

                for feature_func, params in feature_funcs:
                    feat = (
                        feature_func(x_selected)
                        if params is None
                        else feature_func(x_selected, **params)
                    )
                    for state_pos in range(len(state_indices)):
                        feature_values.append(feat[:, state_pos])

                stacked = jnp.stack(feature_values, axis=0)
                return jnp.transpose(stacked, (1, 0))

            if self.use_jit:
                return jax.jit(extract_uniform)  # pyright: ignore[reportUnknownMemberType]
            return extract_uniform

        state_feature_funcs: list[tuple[int, Callable[..., Array], dict[str, object] | None]] = []

        for state_idx, fc_params in self._state_feature_config.items():
            for feature_name, param_list in fc_params.items():
                if feature_name not in ALL_FEATURE_FUNCTIONS:
                    available = list(ALL_FEATURE_FUNCTIONS.keys())
                    raise ValueError(f"Unknown feature: {feature_name}. Available: {available}")

                func = ALL_FEATURE_FUNCTIONS[feature_name]

                if param_list is None:
                    state_feature_funcs.append((state_idx, func, None))
                else:
                    for params in param_list:
                        state_feature_funcs.append((state_idx, func, params))

        def extract_all_features(x: Array) -> Array:
            feature_values: list[Array] = []
            for state_idx, feature_func, params in state_feature_funcs:
                x_state = x[:, :, state_idx : state_idx + 1]
                feat = feature_func(x_state) if params is None else feature_func(x_state, **params)
                feature_values.append(feat.squeeze(-1))

            stacked = jnp.stack(feature_values, axis=0)
            return jnp.transpose(stacked, (1, 0))

        if self.use_jit:
            return jax.jit(extract_all_features)  # pyright: ignore[reportUnknownMemberType]
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
    def feature_names(self) -> list[str]:
        """Return the list of feature names in the format 'state_X__feature_name'."""
        if self._state_feature_config is None or self._num_states is None:
            raise RuntimeError(
                "Feature configuration not initialized. Call extract_features first."
            )

        names: list[str] = []
        for state_idx, fc_params in self._state_feature_config.items():
            for fname in get_feature_names_from_config(fc_params):
                names.append(f"state_{state_idx}__{fname}")
        return names
