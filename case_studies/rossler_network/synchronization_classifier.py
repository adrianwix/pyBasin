"""Synchronization classifier for network basin stability.

This module provides a classifier that determines whether a network of oscillators
has synchronized based on the synchronization features (max deviation).
"""

import numpy as np

from pybasin.predictors.base import LabelPredictor


class SynchronizationClassifier(LabelPredictor):
    """
    Classifier that labels trajectories as 'synchronized' or 'desynchronized'.

    Works with features from SynchronizationFeatureExtractor, which computes
    the max deviation across all node pairs. Synchronization is achieved when:
        max_deviation_all < epsilon

    Parameters
    ----------
    epsilon : float
        Synchronization threshold. States are synchronized if max deviation < epsilon.
    feature_index : int
        Index of the feature to use for thresholding. Default is 3 (max_deviation_all).
        Options: 0=max_deviation_x, 1=max_deviation_y, 2=max_deviation_z, 3=max_deviation_all
    """

    display_name = "Synchronization Classifier"

    def __init__(
        self,
        epsilon: float = 0.1,
        feature_index: int = 3,
    ):
        self.epsilon = epsilon
        self.feature_index = feature_index

    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        """
        Classify each trajectory as synchronized or desynchronized.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix from SynchronizationFeatureExtractor.
            Shape: (n_samples, 4) with columns [max_dev_x, max_dev_y, max_dev_z, max_dev_all]

        Returns
        -------
        np.ndarray
            Labels: 'synchronized' or 'desynchronized' for each trajectory.
        """
        max_deviation = features[:, self.feature_index]

        labels = np.where(
            max_deviation < self.epsilon,
            "synchronized",
            "desynchronized",
        )

        return labels
