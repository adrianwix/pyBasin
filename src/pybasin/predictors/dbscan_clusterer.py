from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from pybasin.predictors.base import ClustererPredictor


class DBSCANClusterer(ClustererPredictor):
    """DBSCAN clustering for basin stability analysis (unsupervised learning)."""

    display_name: str = "DBSCAN Clustering"

    clusterer: DBSCAN

    def __init__(self, clusterer: DBSCAN | None = None, **kwargs: Any):
        """
        Initialize DBSCAN clusterer.

        :param clusterer: DBSCAN instance, or None to create default.
        :param kwargs: Additional arguments for DBSCAN if clusterer is None.
        """
        if clusterer is None:
            clusterer = DBSCAN(**kwargs)
        self.clusterer = clusterer

    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels using DBSCAN clustering.

        :param features: Feature array to cluster.
        :return: Cluster labels.
        """
        return self.clusterer.fit_predict(features)
