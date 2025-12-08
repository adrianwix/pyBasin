"""Feature extractors for time series analysis."""

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.feature_extractors.torch_feature_extractor import TorchFeatureExtractor

__all__ = [
    "FeatureExtractor",
    "TorchFeatureExtractor",
]
