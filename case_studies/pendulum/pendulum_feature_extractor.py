import torch

from pybasin.feature_extractor import FeatureExtractor
from pybasin.solution import Solution

PendulumOHE = {"FP": [1, 0], "LC": [0, 1]}


class PendulumFeatureExtractor(FeatureExtractor):
    def extract_features(self, solution: Solution):
        y_filtered = self.filter_time(solution)  # shape: (N_after, B, S)

        # shape: (N_after, B)
        angular_velocity = y_filtered[..., 1]
        delta = (angular_velocity.max(dim=0).values - angular_velocity.mean(dim=0)).abs()
        mask = delta < 0.01

        fp_tensor = torch.tensor(PendulumOHE["FP"], dtype=torch.float64)  # [1, 0]
        lc_tensor = torch.tensor(PendulumOHE["LC"], dtype=torch.float64)  # [0, 1]

        # shape: (B, 2)
        out = torch.empty((angular_velocity.shape[1], 2), dtype=torch.float64)
        out[mask] = fp_tensor
        out[~mask] = lc_tensor
        return out
