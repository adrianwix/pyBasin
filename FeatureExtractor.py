import copy
from abc import ABC, abstractmethod
import torch
from Solution import Solution


class FeatureExtractor(ABC):

    def __init__(self, time_steady: float, exclude_states=None):
        self.exclude_states = exclude_states
        # defaults to zero - extract feature from complete time series. Recommended
        self.time_steady = time_steady
        # choice however is last 10% of the time signal to avoid transients

    @abstractmethod
    def extract_features(self, solution: Solution) -> torch.Tensor:
        pass

    def filter_states(self, solution: Solution) -> torch.Tensor:
        # solution.y: (N, B, S) => time, batch, states
        # We're removing certain state indices across the last dimension
        if self.exclude_states is not None:
            all_indices = torch.arange(solution.y.shape[2])
            mask = torch.ones_like(all_indices, dtype=torch.bool)
            for idx in self.exclude_states:
                mask[idx] = False
            y_filtered = solution.y[..., mask]
        else:
            y_filtered = solution.y
        return y_filtered

    def filter_time(self, solution: Solution) -> torch.Tensor:
        # solution.time: (N,); solution.y: (N, B, S)
        # We filter along the first dimension (time)
        time_arr = solution.time
        idx_steady = torch.where(time_arr > self.time_steady)[0]
        start_idx = idx_steady[0] if len(idx_steady) > 0 else 0
        y_filtered = solution.y[start_idx:, ...]  # slices time dimension first
        return y_filtered


PendulumOHE = {"FP": [1, 0], "LC": [0, 1]}

# One-hot encoding for Lorenz attractor states
# S1: positive x-mean, S2: negative x-mean
LorenzOHE = {"S1": [1, 0], "S2": [0, 1]}


class PendulumFeatureExtractor(FeatureExtractor):

    def extract_features(self, solution: Solution):
        y_filtered = self.filter_time(solution)  # shape: (N_after, B, S)

        # shape: (N_after, B)
        angular_velocity = y_filtered[..., 1]
        delta = (angular_velocity.max(dim=0).values -
                 angular_velocity.mean(dim=0)).abs()
        mask = delta < 0.01

        fp_tensor = torch.tensor(
            PendulumOHE["FP"], dtype=torch.float64)  # [1, 0]
        lc_tensor = torch.tensor(
            PendulumOHE["LC"], dtype=torch.float64)  # [0, 1]

        # shape: (B, 2)
        out = torch.empty((angular_velocity.shape[1], 2), dtype=torch.float64)
        out[mask] = fp_tensor
        out[~mask] = lc_tensor
        return out


class DuffingFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for the Duffing oscillator.
    Extracts two features from the first state variable:
    1. Maximum value
    2. Standard deviation
    """

    def extract_features(self, solution: Solution):
        y_filtered = self.filter_time(solution)
        # Index 0 corresponds to the first state variable
        max_val = torch.max(y_filtered[..., 0])
        std_val = torch.std(y_filtered[..., 0])
        return torch.tensor([max_val, std_val], dtype=torch.float64)


class LorenzFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for the Lorenz system.
    Classifies trajectories based on the mean x-coordinate:
    - If mean(x) > 0: Solution 1 (positive regime)
    - If mean(x) < 0: Solution 2 (negative regime)
    Also handles unbounded solutions (|y| > 195)
    """

    def extract_features(self, solution: Solution) -> torch.Tensor:
        y_filtered = self.filter_time(solution)  # shape: (N_after, B, S)

        # Check for unbounded solutions first
        max_abs_vals = torch.max(torch.abs(y_filtered), dim=0)[
            0]  # shape: (B, S)
        unbounded = torch.any(max_abs_vals > 195, dim=1)  # shape: (B,)

        # Calculate mean x-coordinate (first state variable)
        x_mean = y_filtered[..., 0].mean(dim=0)  # shape: (B,)
        positive_regime = x_mean > 0

        s1_tensor = torch.tensor(
            LorenzOHE["S1"], dtype=torch.float64)  # [1, 0]
        s2_tensor = torch.tensor(
            LorenzOHE["S2"], dtype=torch.float64)  # [0, 1]

        # Initialize output tensor
        out = torch.empty((y_filtered.shape[1], 2), dtype=torch.float64)

        # Assign features based on x_mean and boundedness
        out[~unbounded & positive_regime] = s1_tensor
        out[~unbounded & ~positive_regime] = s2_tensor
        # Unbounded solutions get [0, 0]
        out[unbounded] = torch.zeros_like(s1_tensor)

        return out
