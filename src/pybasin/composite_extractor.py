"""Composite feature extractor for combining multiple extractors."""

import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution


class CompositeExtractor(FeatureExtractor):
    """Composite feature extractor that concatenates outputs of multiple extractors.

    This allows combining different types of features (e.g., time-domain statistics,
    Lyapunov exponents, correlation dimensions) into a single feature vector.

    Args:
        extractors: List of FeatureExtractor instances to combine.
        time_steady: Time threshold for filtering transients. Default 0.0.
            Note: Individual extractors can override this with their own time_steady.

    Example:
        >>> from pybasin.jax_feature_extractor import JaxFeatureExtractor
        >>> from pybasin.nolds_feature_extractor import LyapunovFeatureExtractor
        >>>
        >>> composite = CompositeExtractor([
        ...     JaxFeatureExtractor(time_steady=950.0),
        ...     LyapunovFeatureExtractor(time_steady=950.0),
        ... ])
        >>> features = composite.extract_features(solution)  # (B, F_jax + F_lyap)
    """

    def __init__(
        self,
        extractors: list[FeatureExtractor],
        time_steady: float = 0.0,
    ):
        super().__init__(time_steady=time_steady)
        self.extractors = extractors

    def extract_features(self, solution: Solution) -> torch.Tensor:
        """Extract and concatenate features from all extractors.

        Args:
            solution: ODE solution with shape (N, B, S)

        Returns:
            Concatenated features tensor of shape (B, F_total) where
            F_total = sum(F_i) for each extractor i.
        """
        if not self.extractors:
            raise ValueError("CompositeExtractor requires at least one extractor")

        # Extract features from each extractor
        feature_list = [extractor.extract_features(solution) for extractor in self.extractors]

        # Ensure all features are on the same device (use device of first feature tensor)
        target_device = feature_list[0].device
        feature_list = [f.to(target_device) for f in feature_list]

        # Concatenate along feature dimension (axis=1)
        features = torch.cat(feature_list, dim=1)

        return features
