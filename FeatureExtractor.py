import copy
from abc import ABC, abstractmethod
import numpy as np
from Solution import Solution


class FeatureExtractor(ABC):

    def __init__(self, time_steady: float, exclude_states=None):
        self.exclude_states = exclude_states
        self.time_steady = time_steady  # defaults to zero - extract feature from complete time series. Recommended
        # choice however is last 10% of the time signal to avoid transients

    @abstractmethod
    def extract_features(self, solution: Solution):
        pass

    # TODO: Find out where do we need this. 
    def filter_states(self, solution: Solution):
        # exclude states (self.exclude_states is a tuple)

        if self.exclude_states is not None:
            y_filtered = np.delete(solution.y, self.exclude_states, axis=1)
        else:
            y_filtered = solution.y

        return y_filtered

    def filter_time(self, solution: Solution):
        # extract the time section for which the user wants to compute the feature
        if self.time_steady is None:
            raise ValueError("time_steady is not set")
        
        idx_steady_state = np.where(solution.time > self.time_steady)[0]
        idx_start = idx_steady_state[0]

        y_filtered = solution.trajectory[idx_start:, :]

        return y_filtered


PendulumOHE = {"FP": [1, 0], "LC": [0, 1]}

class PendulumFeatureExtractor(FeatureExtractor):

    def extract_features(self, solution: Solution):
        temp_sol = copy.deepcopy(solution)

        # get the steady-state time regime
        temp_sol.y = self.filter_time(solution=solution)

        # exclude some states
        temp_sol.y = self.filter_states(solution=temp_sol)

        # only theta_dot is used to calculate delta
        delta = np.abs(np.max(temp_sol.y[:, 1]) - np.mean(temp_sol.y[:, 1]))
        print(f"Delta = {delta}")

        if delta < 0.01:
            print("Fixed Point (FP)")
            # FP (Fixed Point)
            return np.array(PendulumOHE["FP"], dtype=np.float64)
        else:
            # LC (Limit Cycle)
            return np.array(PendulumOHE["LC"], dtype=np.float64)

