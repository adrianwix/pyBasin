from abc import ABC, abstractmethod
import torch
from pybasin.Solution import Solution


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
