"""Abstract base class for extracting features from ODE solution trajectories."""

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
