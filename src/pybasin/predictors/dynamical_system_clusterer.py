# pyright: basic
"""Dynamical system clusterer for two-stage attractor type classification."""

import numpy as np
from sklearn.preprocessing import StandardScaler

from pybasin.predictors.base import ClustererPredictor, LabelPredictor
from pybasin.predictors.hdbscan_clusterer import HDBSCANClusterer


class DynamicalSystemClusterer(ClustererPredictor):
    """Two-stage hierarchical clustering for dynamical systems.

    Stage 1: Classify trajectories by attractor TYPE using physics-based thresholds:
        - Fixed Point (FP): Very low variance in steady state
        - Limit Cycle (LC): Periodic oscillations with high autocorrelation periodicity
        - Chaos: Aperiodic oscillations (not FP, not LC)

    Stage 2: Sub-classify within each type using specialized features and sub-classifiers.

    Required features (must be present in feature_names):
        - variance: For FP detection
        - amplitude: For LC amplitude distinction
        - mean: For FP/attractor location
        - linear_trend__attr_slope: For drift/rotation detection
        - autocorrelation_periodicity__output_strength: For LC vs chaos detection
        - autocorrelation_periodicity__output_period: For period estimation
        - spectral_frequency_ratio: For period-1/2/3 LC distinction
    """

    display_name: str = "Dynamical System Clustering"

    REQUIRED_FEATURES = [
        "variance",
        "amplitude",
        "mean",
        "linear_trend__attr_slope",
        "autocorrelation_periodicity__output_strength",
        "autocorrelation_periodicity__output_period",
        "spectral_frequency_ratio",
    ]

    def __init__(
        self,
        feature_names: list[str],
        fp_variance_threshold: float = 1e-6,
        lc_periodicity_threshold: float = 0.5,
        chaos_variance_threshold: float = 5.0,
        drift_threshold: float = 0.1,
        tiers: list[str] | None = None,
        fp_sub_classifier: LabelPredictor | None = None,
        lc_sub_classifier: LabelPredictor | None = None,
        chaos_sub_classifier: LabelPredictor | None = None,
    ):
        """Initialize the dynamical system clusterer.

        Args:
            feature_names: List of feature names matching the feature array columns.
            fp_variance_threshold: Maximum variance to classify as fixed point.
            lc_periodicity_threshold: Minimum periodicity strength for limit cycle.
            chaos_variance_threshold: Maximum variance for limit cycle (above = chaos).
            drift_threshold: Minimum slope magnitude to consider as drifting.
            tiers: List of attractor types to detect. Default: ["FP", "LC", "chaos"].
                   Order matters: first match wins.
            fp_sub_classifier: Sub-classifier for fixed points. Default: KMeans.
            lc_sub_classifier: Sub-classifier for limit cycles. Default: HDBSCAN.
            chaos_sub_classifier: Sub-classifier for chaotic attractors. Default: HDBSCAN.
        """
        self.feature_names = feature_names
        self.fp_variance_threshold = fp_variance_threshold
        self.lc_periodicity_threshold = lc_periodicity_threshold
        self.chaos_variance_threshold = chaos_variance_threshold
        self.drift_threshold = drift_threshold
        self.tiers = tiers or ["FP", "LC", "chaos"]

        self.fp_sub_classifier = fp_sub_classifier
        self.lc_sub_classifier = lc_sub_classifier
        self.chaos_sub_classifier = chaos_sub_classifier

        self._feature_indices: dict[str, list[int]] = {}
        self._drifting_dims: list[int] = []
        self._non_drifting_dims: list[int] = []

        self._build_feature_indices(feature_names)

    def _build_feature_indices(self, feature_names: list[str]) -> None:
        """Build mapping from feature base names to column indices.

        Handles multi-dimensional features with naming convention:
            - state_X__feature_name -> single value
            - state_X__feature_name__idx -> multi-value (e.g., autocorrelation_periodicity)

        Args:
            feature_names: List of feature names.

        Raises:
            ValueError: If required features are missing.
        """
        self._feature_indices = {}

        for base_feature in self.REQUIRED_FEATURES:
            matching_indices = []
            for idx, name in enumerate(feature_names):
                parts = name.split("__")
                if len(parts) >= 2:
                    feature_part = "__".join(parts[1:])
                    if feature_part == base_feature or feature_part.startswith(base_feature + "__"):
                        matching_indices.append(idx)

            if not matching_indices:
                raise ValueError(
                    f"Required feature '{base_feature}' not found in feature_names. "
                    f"Available: {feature_names}"
                )

            self._feature_indices[base_feature] = matching_indices

    def _detect_drifting_dims(self, features: np.ndarray) -> None:
        """Detect dimensions with linear drift (rotating variables).

        Uses linear_trend__attr_slope to identify dimensions where the majority
        of trajectories show monotonic drift (e.g., pendulum angle for rotating solutions).

        Sets self._drifting_dims and self._non_drifting_dims.

        Args:
            features: Full feature array (n_samples, n_features).
        """
        slope_indices = self._feature_indices["linear_trend__attr_slope"]
        n_dims = len(slope_indices)

        self._drifting_dims = []
        for dim_idx, slope_idx in enumerate(slope_indices):
            slopes = features[:, slope_idx]
            high_slope_fraction = np.mean(np.abs(slopes) > self.drift_threshold)
            if high_slope_fraction > 0.3:
                self._drifting_dims.append(dim_idx)

        self._non_drifting_dims = [d for d in range(n_dims) if d not in self._drifting_dims]
        if not self._non_drifting_dims:
            self._non_drifting_dims = list(range(n_dims))

    def _get_feature_values(
        self, features: np.ndarray, base_feature: str, state_idx: int = 0
    ) -> np.ndarray:
        """Get feature values for a specific base feature.

        Args:
            features: Full feature array (n_samples, n_features).
            base_feature: Base feature name (e.g., "variance").
            state_idx: Which state's feature to use (for multi-state systems).

        Returns:
            Feature values array of shape (n_samples,).
        """
        indices = self._feature_indices[base_feature]
        if state_idx < len(indices):
            return features[:, indices[state_idx]]
        return features[:, indices[0]]

    def _classify_attractor_type(self, features: np.ndarray) -> np.ndarray:
        """Classify trajectories by attractor type (Stage 1).

        Args:
            features: Feature array (n_samples, n_features).

        Returns:
            Array of type labels: "FP", "LC", "chaos".
        """
        n_samples = features.shape[0]
        type_labels = np.empty(n_samples, dtype=object)
        type_labels[:] = "chaos"

        variance_indices = self._feature_indices["variance"]
        non_drifting_var_indices = [
            variance_indices[d] for d in self._non_drifting_dims if d < len(variance_indices)
        ]
        if not non_drifting_var_indices:
            non_drifting_var_indices = variance_indices
        variance = np.mean(features[:, non_drifting_var_indices], axis=1)

        periodicity_strength = self._get_feature_values(
            features, "autocorrelation_periodicity__output_strength"
        )
        drift_slope = self._get_feature_values(features, "linear_trend__attr_slope")

        for tier in self.tiers:
            if tier == "FP":
                fp_mask = (variance < self.fp_variance_threshold) & (type_labels == "chaos")
                type_labels[fp_mask] = "FP"

            elif tier == "LC":
                has_periodicity = periodicity_strength > self.lc_periodicity_threshold
                not_high_variance = variance < self.chaos_variance_threshold
                has_drift = np.abs(drift_slope) > self.drift_threshold
                lc_periodic = has_periodicity & not_high_variance
                lc_rotating = has_drift & (variance > self.fp_variance_threshold)
                lc_mask = (lc_periodic | lc_rotating) & (type_labels == "chaos")
                type_labels[lc_mask] = "LC"

        return type_labels

    def _sub_classify_fixed_points(
        self,
        features: np.ndarray,
        indices: np.ndarray,
        single_cluster_range_threshold: float = 0.01,
    ) -> np.ndarray:
        """Sub-classify fixed points by steady-state location.

        Excludes drifting dimensions from clustering to avoid spurious splits.

        Args:
            features: Full feature array.
            indices: Indices of FP trajectories.
            single_cluster_range_threshold: If range (max-min) of mean values
                is below this threshold for all dimensions, treat as single cluster.

        Returns:
            Sub-cluster labels for FP trajectories.
        """
        if len(indices) == 0:
            return np.array([], dtype=int)

        mean_indices = self._feature_indices["mean"]
        non_drifting_mean_indices = [
            mean_indices[d] for d in self._non_drifting_dims if d < len(mean_indices)
        ]
        if not non_drifting_mean_indices:
            non_drifting_mean_indices = mean_indices

        fp_features = features[indices][:, non_drifting_mean_indices]

        if self.fp_sub_classifier is not None:
            return self.fp_sub_classifier.predict_labels(fp_features)

        if len(indices) < 2:
            return np.zeros(len(indices), dtype=int)

        data_range = np.max(fp_features, axis=0) - np.min(fp_features, axis=0)
        if np.all(data_range < single_cluster_range_threshold):
            return np.zeros(len(indices), dtype=int)

        scaler = StandardScaler()
        fp_scaled = scaler.fit_transform(fp_features)

        min_cluster_size = max(50, len(indices) // 10)
        clusterer = HDBSCANClusterer(
            min_cluster_size=min_cluster_size, auto_tune=False, assign_noise=True
        )
        labels = clusterer.predict_labels(fp_scaled)
        return labels

    def _sub_classify_limit_cycles(
        self,
        features: np.ndarray,
        indices: np.ndarray,
        amp_cv_threshold: float = 0.1,
        mean_range_threshold: float = 0.05,
    ) -> np.ndarray:
        """Sub-classify limit cycles using hierarchical period-based approach.

        Two-level hierarchical clustering:
        1. First level: Group by period-n (freq_ratio rounded to nearest integer)
        2. Second level: Within each period group, cluster by amplitude and mean

        Uses non-drifting dimensions for amplitude and mean features.

        Args:
            features: Full feature array.
            indices: Indices of LC trajectories.
            amp_cv_threshold: Min coefficient of variation for amplitude to cluster.
            mean_range_threshold: Min range for mean to cluster.

        Returns:
            Sub-cluster labels for LC trajectories.
        """
        if len(indices) == 0:
            return np.array([], dtype=int)

        analysis_dim = self._non_drifting_dims[0] if self._non_drifting_dims else 0

        freq_ratio_indices = self._feature_indices["spectral_frequency_ratio"]
        amp_indices = self._feature_indices["amplitude"]
        mean_indices = self._feature_indices["mean"]

        freq_ratio_idx = (
            freq_ratio_indices[analysis_dim]
            if analysis_dim < len(freq_ratio_indices)
            else freq_ratio_indices[0]
        )
        amp_idx = amp_indices[analysis_dim] if analysis_dim < len(amp_indices) else amp_indices[0]
        mean_idx = (
            mean_indices[analysis_dim] if analysis_dim < len(mean_indices) else mean_indices[0]
        )

        freq_ratios = features[indices, freq_ratio_idx]
        amplitudes = features[indices, amp_idx]
        means = features[indices, mean_idx]

        if self.lc_sub_classifier is not None:
            lc_features = features[indices][:, [freq_ratio_idx, amp_idx, mean_idx]]
            return self.lc_sub_classifier.predict_labels(lc_features)

        finite_mask = np.isfinite(freq_ratios) & np.isfinite(amplitudes) & np.isfinite(means)
        if not np.any(finite_mask):
            return np.zeros(len(indices), dtype=int)

        period_n = np.round(freq_ratios).astype(int)
        period_n = np.clip(period_n, 1, 10)

        unique_periods = np.unique(period_n[finite_mask])
        labels = np.zeros(len(indices), dtype=int)
        current_label = 0

        for period in unique_periods:
            period_mask = (period_n == period) & finite_mask
            period_indices = np.where(period_mask)[0]

            if len(period_indices) == 0:
                continue

            period_amps = amplitudes[period_mask]
            period_means = means[period_mask]

            amp_cv = np.std(period_amps) / (np.mean(period_amps) + 1e-10)
            mean_range = np.max(period_means) - np.min(period_means)

            amp_varies = amp_cv > amp_cv_threshold
            mean_varies = mean_range > mean_range_threshold

            if not amp_varies and not mean_varies:
                labels[period_mask] = current_label
                current_label += 1
            else:
                if amp_varies and mean_varies:
                    sub_features = np.column_stack([period_amps, period_means])
                elif amp_varies:
                    sub_features = period_amps.reshape(-1, 1)
                else:
                    sub_features = period_means.reshape(-1, 1)

                sub_labels = self._cluster_1d_or_2d(sub_features, len(period_indices))

                for sub_label in np.unique(sub_labels):
                    sub_mask = sub_labels == sub_label
                    full_mask = np.zeros(len(indices), dtype=bool)
                    full_mask[period_indices[sub_mask]] = True
                    labels[full_mask] = current_label
                    current_label += 1

        labels[~finite_mask] = 0

        return labels

    def _cluster_1d_or_2d(self, data: np.ndarray, n_samples: int) -> np.ndarray:
        """Cluster 1D or 2D data using appropriate method.

        For 1D data: detect natural gaps/modes.
        For 2D data: use HDBSCAN.

        Args:
            data: Feature array of shape (n_samples, 1) or (n_samples, 2).
            n_samples: Number of samples.

        Returns:
            Cluster labels.
        """
        if data.shape[1] == 1:
            return self._cluster_1d_with_gaps(data.ravel())
        else:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(data)

            min_cluster_size = max(15, n_samples // 20)
            use_auto_tune = n_samples >= 500
            clusterer = HDBSCANClusterer(
                min_cluster_size=min_cluster_size,
                auto_tune=use_auto_tune,
                assign_noise=True,
            )
            return clusterer.predict_labels(scaled)

    def _cluster_1d_with_gaps(
        self, data: np.ndarray, min_gap_ratio: float = 5.0, max_clusters: int = 5
    ) -> np.ndarray:
        """Cluster 1D data by detecting the largest natural gap(s).

        Finds gaps in sorted data that are significantly larger than typical spacing.
        Only creates up to max_clusters to avoid over-splitting.

        Args:
            data: 1D array of values.
            min_gap_ratio: Minimum ratio of gap to median gap to be significant.
            max_clusters: Maximum number of clusters to create.

        Returns:
            Cluster labels.
        """
        n = len(data)
        if n < 4:
            return np.zeros(n, dtype=int)

        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]

        gaps = np.diff(sorted_data)

        if np.max(gaps) < 1e-6:
            return np.zeros(n, dtype=int)

        gap_threshold = np.median(gaps) * min_gap_ratio
        gap_threshold = max(gap_threshold, np.max(gaps) * 0.3)

        significant_gap_indices = np.where(gaps > gap_threshold)[0]

        if len(significant_gap_indices) == 0:
            return np.zeros(n, dtype=int)

        if len(significant_gap_indices) >= max_clusters:
            gap_values = gaps[significant_gap_indices]
            top_gaps = np.argsort(gap_values)[-(max_clusters - 1) :]
            significant_gap_indices = sorted(significant_gap_indices[top_gaps])

        labels_sorted = np.zeros(n, dtype=int)
        current_label = 0
        for i in range(n - 1):
            labels_sorted[i] = current_label
            if i in significant_gap_indices:
                current_label += 1
        labels_sorted[n - 1] = current_label

        labels = np.zeros(n, dtype=int)
        labels[sorted_indices] = labels_sorted

        return labels

    def _sub_classify_chaos(self, features: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Sub-classify chaotic trajectories by spatial mean.

        Args:
            features: Full feature array.
            indices: Indices of chaos trajectories.

        Returns:
            Sub-cluster labels for chaos trajectories.
        """
        if len(indices) == 0:
            return np.array([], dtype=int)

        mean_indices = self._feature_indices["mean"]
        chaos_features = features[indices][:, mean_indices]

        if self.chaos_sub_classifier is not None:
            return self.chaos_sub_classifier.predict_labels(chaos_features)

        finite_mask = np.all(np.isfinite(chaos_features), axis=1)
        if np.sum(finite_mask) < 2:
            return np.zeros(len(indices), dtype=int)

        scaler = StandardScaler()
        chaos_features_clean = chaos_features.copy()
        chaos_features_clean[~finite_mask] = 0
        chaos_scaled = scaler.fit_transform(chaos_features_clean)

        clusterer = HDBSCANClusterer(auto_tune=True, assign_noise=True)
        labels = clusterer.predict_labels(chaos_scaled)
        return labels

    def predict_labels(self, features: np.ndarray) -> np.ndarray:
        """Predict labels using two-stage hierarchical clustering.

        Args:
            features: Feature array of shape (n_samples, n_features).

        Returns:
            Array of predicted labels with format "TYPE_subcluster".
        """
        self._detect_drifting_dims(features)

        n_samples = features.shape[0]
        type_labels = self._classify_attractor_type(features)

        final_labels = np.empty(n_samples, dtype=object)
        current_label = 0

        for attractor_type in ["FP", "LC", "chaos"]:
            if attractor_type not in self.tiers:
                continue

            type_indices = np.where(type_labels == attractor_type)[0]

            if len(type_indices) == 0:
                continue

            if attractor_type == "FP":
                sub_labels = self._sub_classify_fixed_points(features, type_indices)
            elif attractor_type == "LC":
                sub_labels = self._sub_classify_limit_cycles(features, type_indices)
            else:
                sub_labels = self._sub_classify_chaos(features, type_indices)

            for sub_label in np.unique(sub_labels):
                mask = sub_labels == sub_label
                final_labels[type_indices[mask]] = f"{attractor_type}_{current_label}"
                current_label += 1

        return final_labels
