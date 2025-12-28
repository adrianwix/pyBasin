import numpy as np
from sklearn.cluster import DBSCAN

from pybasin.predictors.dbscan_clusterer import DBSCANClusterer


def test_dbscan_clusterer():
    dbscan = DBSCANClusterer(clusterer=DBSCAN(eps=0.5, min_samples=2))

    features = np.array([[0, 0], [0.1, 0.1], [5, 5], [5.1, 5]])
    labels = dbscan.predict_labels(features)

    # Four labels (one per feature point)
    assert len(labels) == 4
    # At least 2 clusters found (two pairs of nearby points)
    assert len(set(labels)) >= 2
