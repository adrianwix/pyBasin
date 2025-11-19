"""Abstract base class for extracting features from ODE solution trajectories."""

from abc import ABC, abstractmethod

import torch

from pybasin.solution import Solution


# TODO: Review. I am not entirely convinced with this implementation. It could be made more customizable.
class FeatureExtractor(ABC):
    """Abstract base class for extracting features from ODE solutions.

    Feature extractors transform ODE solution trajectories into feature vectors
    that can be used for basin of attraction classification. This class provides
    utilities for filtering solutions by time (to remove transients) and by state
    (to exclude certain state variables).

    Args:
        time_steady: Time threshold for filtering transients. Only data after this
            time will be used for feature extraction. Default of 0.0 uses the entire
            time series. A common choice is the last 10% of the integration time to
            avoid transient behavior.
        exclude_states: Optional list of state indices to exclude from feature
            extraction. Useful when certain state variables are not relevant for
            classification (e.g., time-dependent forcing terms).

    Example:
        >>> class AmplitudeExtractor(FeatureExtractor):
        ...     def extract_features(self, solution: Solution) -> torch.Tensor:
        ...         y_filtered = self.filter_time(solution)
        ...         y_filtered = self.filter_states(solution)
        ...         return torch.max(torch.abs(y_filtered), dim=0)[0]
    """

    def __init__(self, time_steady: float = 0.0, exclude_states: list[int] | None = None):
        self.exclude_states = exclude_states
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

    def filter_states(self, solution: Solution) -> torch.Tensor:
        """Filter out excluded state variables from the solution.

        Removes state variables specified in `exclude_states` from the solution
        tensor. This is useful when certain state variables (e.g., auxiliary
        variables, forcing terms) should not contribute to feature extraction.

        Args:
            solution: ODE solution with y tensor of shape (N, B, S) where N is
                time steps, B is batch size, and S is number of state variables.

        Returns:
            Filtered tensor of shape (N, B, S') where S' = S - len(exclude_states)
            is the number of remaining state variables after filtering. If
            exclude_states is None, returns the original solution.y unchanged.

        Example:
            >>> # Exclude the 3rd state variable (index 2)
            >>> extractor = FeatureExtractor(exclude_states=[2])
            >>> filtered = extractor.filter_states(solution)
            >>> # If solution.y had shape (100, 50, 4), filtered has shape (100, 50, 3)
        """
        if self.exclude_states is not None:
            # Create a boolean mask to select which state indices to keep
            all_indices = torch.arange(solution.y.shape[2])
            mask = torch.ones_like(all_indices, dtype=torch.bool)
            for idx in self.exclude_states:
                mask[idx] = False
            # Apply mask to the last dimension (state variables)
            y_filtered = solution.y[..., mask]
        else:
            y_filtered = solution.y
        return y_filtered

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
