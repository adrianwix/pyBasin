"""Feature extractor using tsfresh library for time series feature extraction."""

from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import torch
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features  # type: ignore[import-untyped]
from tsfresh.feature_extraction import MinimalFCParameters  # type: ignore[import-untyped]
from tsfresh.utilities.dataframe_functions import impute  # type: ignore[import-untyped]

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution


class TsfreshFeatureExtractor(FeatureExtractor):
    """Feature extractor using tsfresh for comprehensive time series analysis.

    This extractor uses the tsfresh library to automatically extract a large number
    of time series features from ODE solutions. It converts PyTorch/JAX tensors to
    pandas DataFrames for tsfresh processing, then converts the results back to tensors.

    Supports per-state variable feature configuration using tsfresh's kind_to_fc_parameters
    mechanism, allowing you to apply different feature sets to different state variables
    based on domain knowledge.

    Args:
        time_steady: Time threshold for filtering transients. Only data after this
            time will be used for feature extraction. Default of 0.0 uses the entire
            time series.
        default_fc_parameters: Default feature extraction parameters for all states.
            Can be one of:
            - MinimalFCParameters() - Fast extraction with ~20 features
            - ComprehensiveFCParameters() - Full extraction with ~800 features
            - Custom dict like {"mean": None, "maximum": None} for specific features
            - None - must provide kind_to_fc_parameters
            Default is MinimalFCParameters().
        kind_to_fc_parameters: Optional dict mapping state indices to FCParameters.
            Allows different feature sets per state variable. If provided, overrides
            default_fc_parameters for those states. Example:
            {
                0: {"mean": None, "maximum": None},  # State 0: only mean and max
                1: ComprehensiveFCParameters(),  # State 1: all features
            }
        n_jobs: Number of parallel jobs for feature extraction. Default is 1.
            Set to -1 to use all available cores.
        normalize: Whether to apply StandardScaler normalization to features.
            Highly recommended for distance-based classifiers like KNN.
            Default is True.

    Examples:
        >>> # Same minimal features for all states
        >>> extractor = TsfreshFeatureExtractor(
        ...     time_steady=9.0,
        ...     default_fc_parameters=MinimalFCParameters(),
        ...     n_jobs=-1,
        ...     normalize=True
        ... )

        >>> # Specific features for all states
        >>> extractor = TsfreshFeatureExtractor(
        ...     time_steady=950.0,
        ...     default_fc_parameters={"mean": None, "std": None, "maximum": None},
        ...     n_jobs=-1
        ... )

        >>> # Different features per state (e.g., pendulum: position vs velocity)
        >>> from tsfresh.feature_extraction import MinimalFCParameters, ComprehensiveFCParameters
        >>> extractor = TsfreshFeatureExtractor(
        ...     time_steady=950.0,
        ...     kind_to_fc_parameters={
        ...         0: {"mean": None, "maximum": None, "minimum": None},  # Position: basic stats
        ...         1: ComprehensiveFCParameters(),  # Velocity: full spectral analysis
        ...     },
        ...     n_jobs=1  # Use n_jobs=1 for deterministic results
        ... )

    Note on parallelism:
        Setting n_jobs > 1 enables parallel feature extraction but introduces
        non-determinism due to floating-point arithmetic order. This can cause
        inconsistent classification results. Use n_jobs=1 for reproducible results.

    Note on normalization:
        When normalize=True, the scaler is fitted on the FIRST dataset that calls
        extract_features(). For best results with supervised classifiers:
        - Either set normalize=False (recommended for KNN with few templates)
        - Or call fit_scaler() explicitly with representative data before extraction
    """

    def __init__(
        self,
        time_steady: float = 0.0,
        default_fc_parameters: dict[str, Any] | Any | None = None,
        kind_to_fc_parameters: dict[int, dict[str, Any] | Any] | None = None,
        n_jobs: int = 1,
        normalize: bool = True,
    ):
        import threading

        super().__init__(time_steady=time_steady)
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self._is_fitted = False
        self._fit_lock = threading.Lock()  # Thread-safe fitting

        # tsfresh doesn't handle n_jobs=-1 well, convert to actual number
        if n_jobs == -1:
            import multiprocessing

            self.n_jobs = multiprocessing.cpu_count()
        elif n_jobs < 1:
            self.n_jobs = 1
        else:
            self.n_jobs = n_jobs

        # Set default feature parameters
        if default_fc_parameters is None and kind_to_fc_parameters is None:
            self.default_fc_parameters: dict[str, Any] | Any | None = MinimalFCParameters()
        else:
            self.default_fc_parameters: dict[str, Any] | Any | None = default_fc_parameters

        self.kind_to_fc_parameters: dict[int, dict[str, Any] | Any] | None = kind_to_fc_parameters

    def reset_scaler(self) -> None:
        """Reset the scaler to unfitted state.

        Call this if you need to refit the scaler on different data.
        """
        if self.normalize:
            self.scaler = StandardScaler()
            self._is_fitted = False

    def extract_features(self, solution: Solution) -> torch.Tensor:
        """Extract features from an ODE solution using tsfresh.

        Converts the solution tensor to pandas DataFrame format expected by tsfresh,
        extracts features for each trajectory and state variable, then converts back
        to PyTorch tensor.

        Args:
            solution: ODE solution with y tensor of shape (N, B, S) where N is
                time steps, B is batch size, and S is number of state variables.

        Returns:
            Feature tensor of shape (B, F) where B is the batch size and F is
            the total number of features extracted by tsfresh across all state
            variables.
        """
        # Apply time filtering
        y_filtered = self.filter_time(solution)

        # Get dimensions
        n_timesteps, n_batch, n_states = y_filtered.shape

        # Convert to numpy for pandas compatibility
        y_np = y_filtered.cpu().numpy()

        # Prepare data in tsfresh wide format
        # Each row represents one time point for one trajectory (batch)
        # Columns: [id, time, state_0, state_1, ..., state_S]
        data_list: list[dict[str, Any]] = []

        for batch_idx in range(n_batch):
            for time_idx in range(n_timesteps):
                row: dict[str, Any] = {
                    "id": batch_idx,
                    "time": time_idx,
                }
                for state_idx in range(n_states):
                    row[f"state_{state_idx}"] = y_np[time_idx, batch_idx, state_idx]
                data_list.append(row)

        df_pivot = pd.DataFrame(data_list)

        # Build kind_to_fc_parameters if per-state config is provided
        if self.kind_to_fc_parameters is not None:
            # Map state column names to their feature parameters
            kind_to_fc_params_mapped = {
                f"state_{state_idx}": fc_params
                for state_idx, fc_params in self.kind_to_fc_parameters.items()
            }
            # Add a column_kind to the dataframe to enable kind-based extraction
            # First, melt to long format with a "kind" column
            id_vars: list[str] = ["id", "time"]
            df_long: pd.DataFrame = df_pivot.melt(  # type: ignore[reportUnknownMemberType]
                id_vars=id_vars, var_name="kind", value_name="value"
            )

            # Extract features using tsfresh with kind_to_fc_parameters
            features_df: pd.DataFrame = cast(
                pd.DataFrame,
                extract_features(
                    df_long,
                    column_id="id",
                    column_sort="time",
                    column_kind="kind",
                    column_value="value",
                    kind_to_fc_parameters=kind_to_fc_params_mapped,
                    n_jobs=self.n_jobs,
                    disable_progressbar=True,
                ),
            )
        else:
            # Extract features using tsfresh with default parameters for all states
            features_df: pd.DataFrame = cast(
                pd.DataFrame,
                extract_features(
                    df_pivot,
                    column_id="id",
                    column_sort="time",
                    default_fc_parameters=self.default_fc_parameters,
                    n_jobs=self.n_jobs,
                    disable_progressbar=True,
                ),
            )

        # Handle NaN and inf values using tsfresh's impute function
        # This replaces NaN with 0 and inf with large finite values
        impute(features_df)

        # Convert to numpy array
        features_array = features_df.values

        # Apply normalization if enabled (thread-safe)
        if self.normalize and self.scaler is not None:
            with self._fit_lock:
                if not self._is_fitted:
                    # Fit and transform on first call
                    features_array = self.scaler.fit_transform(features_array)  # type: ignore[reportUnknownMemberType]
                    self._is_fitted = True
                else:
                    # Transform only on subsequent calls
                    features_array = self.scaler.transform(features_array)
                    # Transform only on subsequent calls (e.g., test data)
                    features_array = self.scaler.transform(features_array)

        # Convert back to tensor
        features_tensor = torch.tensor(features_array, dtype=torch.float32)

        return features_tensor
