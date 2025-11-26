import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution


class MeanFeatureExtractor(FeatureExtractor):
    def extract_features(self, solution: Solution) -> torch.Tensor:
        y_filtered = self.filter_time(solution)
        return y_filtered.mean(dim=0)


def test_feature_extractor_time_filtering():
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    time_steady = 5.0
    extractor = MeanFeatureExtractor(time_steady=time_steady)
    y_filtered = extractor.filter_time(solution)

    # Calculate expected number of time steps after filtering
    expected_steps = (time > time_steady).sum().item()

    # Time dimension should be reduced (only times > 5.0 remain)
    assert y_filtered.shape[0] == expected_steps
    # Batch and state dimensions should remain unchanged
    assert y_filtered.shape[1:] == (n_batch, n_states)


def test_feature_extractor_extract():
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = MeanFeatureExtractor(time_steady=0)
    features = extractor.extract_features(solution)

    # Features should collapse time dimension (mean over time)
    assert features.shape == (n_batch, n_states)
