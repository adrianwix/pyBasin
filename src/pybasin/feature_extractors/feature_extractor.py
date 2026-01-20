"""Abstract base class for extracting features from ODE solution trajectories."""

import re
from abc import ABC, abstractmethod

import torch

from pybasin.solution import Solution


class FeatureExtractor(ABC):
    """Abstract base class for extracting features from ODE solutions.

    Feature extractors transform ODE solution trajectories into feature vectors
    that can be used for basin of attraction classification. This class provides
    utilities for filtering solutions by time (to remove transients).

    Args:
        time_steady: Time threshold for filtering transients. Only data after this
            time will be used for feature extraction. Default of 0.0 uses the entire
            time series. A common choice is the last 10% of the integration time to
            avoid transient behavior.

    Example:
        >>> class AmplitudeExtractor(FeatureExtractor):
        ...     def extract_features(self, solution: Solution) -> torch.Tensor:
        ...         y_filtered = self.filter_time(solution)
        ...         return torch.max(torch.abs(y_filtered), dim=0)[0]
    """

    def __init__(self, time_steady: float = 0.0):
        self.time_steady = time_steady
        self._feature_names: list[str] | None = None
        self._num_features: int | None = None

    @abstractmethod
    def extract_features(self, solution: Solution) -> torch.Tensor:
        """Extract features from an ODE solution.

        This method must be implemented by subclasses to define how features
        are computed from solution trajectories.

        Args:
            solution: ODE solution containing time series data for one or more
                trajectories. The solution.y tensor has shape (N, B, S) where
                N is the number of time steps, B is the batch size (number of
                initial conditions), and S is the number of state variables.

        Returns:
            Feature tensor of shape (B, F) where B is the batch size and F is
            the number of features extracted per trajectory.
        """
        pass

    @property
    def feature_names(self) -> list[str]:
        """Return the list of feature names.

        If not explicitly set by a subclass, automatically generates names using
        the pattern: <class_name_snake_case>_<index>. The class name is converted
        to snake_case and the suffix 'FeatureExtractor' is removed (if present).

        Returns:
            List of feature names. Length must match the number of features (F)
            in the output tensor from extract_features().
        """
        if self._feature_names is not None:
            return self._feature_names

        if self._num_features is None:
            return []

        class_name = self.__class__.__name__
        snake_case_name = self._to_snake_case(class_name)

        if snake_case_name.endswith("_feature_extractor") and len(snake_case_name) > len(
            "_feature_extractor"
        ):
            snake_case_name = snake_case_name[: -len("_feature_extractor")]
        elif snake_case_name == "feature_extractor":
            snake_case_name = "feature"

        if len(snake_case_name) == 0:
            snake_case_name = "feature"

        return [f"{snake_case_name}_{i + 1}" for i in range(self._num_features)]

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert CamelCase to snake_case."""
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def filter_time(self, solution: Solution) -> torch.Tensor:
        """Filter out transient behavior by removing early time steps.

        Removes time steps before `time_steady` to exclude transient dynamics
        from feature extraction. This ensures features are computed only from
        steady-state or long-term behavior.

        Args:
            solution: ODE solution with time tensor of shape (N,) and y tensor
                of shape (N, B, S) where N is time steps, B is batch size, and
                S is number of state variables.

        Returns:
            Filtered tensor of shape (N', B, S) where N' is the number of time
            steps after time_steady. If time_steady is 0 or less than all time
            points, returns the original solution.y unchanged.

        Example:
            >>> # Extract features only from the last 10% of integration time
            >>> extractor = FeatureExtractor(time_steady=9.0)  # if time_span=(0, 10)
            >>> filtered = extractor.filter_time(solution)
            >>> # Only time points t > 9.0 are included
        """
        time_arr = solution.time
        # Find indices where time exceeds the steady-state threshold
        idx_steady = torch.where(time_arr > self.time_steady)[0]
        # Use first qualifying index, or 0 if none found (include all)
        start_idx = idx_steady[0] if len(idx_steady) > 0 else 0
        # Slice along time dimension (first dimension)
        y_filtered = solution.y[start_idx:, ...]
        return y_filtered
