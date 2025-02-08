import copy
from abc import ABC, abstractmethod
import numpy as np
from Solution import Solution


class FeatureExtractor(ABC):

    def __init__(self, time_steady: float, exclude_states=None):
        self.exclude_states = exclude_states
        # defaults to zero - extract feature from complete time series. Recommended
        self.time_steady = time_steady
        # choice however is last 10% of the time signal to avoid transients

    @abstractmethod
    def extract_features(self, solution: Solution):
        pass

    def filter_states(self, solution: Solution):
        # exclude states (self.exclude_states is a tuple)
        if self.exclude_states is not None:
            y_filtered = np.delete(solution.y, self.exclude_states, axis=1)
        else:
            y_filtered = solution.y
        return y_filtered

    def filter_time(self, solution: Solution):
        # Convert time array to numpy if needed, then find indices
        time_arr = solution.time  # shape (T,)
        idx_steady = np.where(time_arr > self.time_steady)[0]
        start_idx = idx_steady[0]
        y_filtered = solution.y[start_idx:, :]
        return y_filtered


PendulumOHE = {"FP": [1, 0], "LC": [0, 1]}


class PendulumFeatureExtractor(FeatureExtractor):

    def extract_features(self, solution: Solution):
        # Use filter_time to get steady-state data
        y_filtered = self.filter_time(solution)
        # y_filtered shape is (T_after, 2)
        delta = np.abs(np.max(y_filtered[:, 1]) - np.mean(y_filtered[:, 1]))

        if delta < 0.01:
            # FP (Fixed Point)
            return np.array(PendulumOHE["FP"], dtype=np.float64)
        else:
            # LC (Limit Cycle)
            return np.array(PendulumOHE["LC"], dtype=np.float64)


class DuffingFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for the Duffing oscillator.
    Extracts two features from the first state variable:
    1. Maximum value
    2. Standard deviation
    """

    def extract_features(self, solution: Solution):
        # Get the steady-state portion of the trajectory
        y_filtered = self.filter_time(solution)

        # Calculate features using first state (y[:,0])
        max_val = np.max(y_filtered[:, 0])
        std_val = np.std(y_filtered[:, 0])

        return np.array([max_val, std_val], dtype=np.float64)
