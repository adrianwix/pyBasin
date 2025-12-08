"""State space statistics feature extractor for trajectory classification."""

from typing import cast

import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution


class StateSpaceStatsExtractor(FeatureExtractor):
    """Extract multi-dimensional statistical features from state space trajectories.

    Focuses on relationships between state dimensions that per-coordinate statistics cannot capture:
    - Mean radius: captures attractor extent in phase space
    - Max radius: captures attractor size
    - PCA eigenvalues: describes anisotropy and effective dimensionality

    Args:
        time_steady: Time threshold for filtering transients. Default 0.0.
        include_radius: Include mean and max radius from centroid. Default True.
        include_pca: Include PCA eigenvalues of covariance matrix. Default True.
    """

    def __init__(
        self,
        time_steady: float = 0.0,
        include_radius: bool = True,
        include_pca: bool = True,
    ):
        super().__init__(time_steady=time_steady)
        self.include_radius = include_radius
        self.include_pca = include_pca
        self._num_states: int | None = None

    def extract_features(self, solution: Solution) -> torch.Tensor:
        """Extract state space statistical features.

        Args:
            solution: ODE solution with shape (N, B, S)

        Returns:
            Features tensor of shape (B, F) where F depends on which features
            are enabled. Features are concatenated in order:
            [mean_radius, max_radius, pca_eigenvalues]
        """
        # Filter time
        y_filtered = self.filter_time(solution)

        # y_filtered shape: (N, B, S)
        # N = time steps, B = batch size, S = state dimensions
        self._num_states = y_filtered.shape[2]

        feature_list: list[torch.Tensor] = []

        # 1. Radius measures
        if self.include_radius:
            mean_states: torch.Tensor = torch.mean(y_filtered, dim=0)  # (B, S)
            # Compute distances from mean: (N, B)
            distances: torch.Tensor = cast(
                torch.Tensor,
                torch.norm(y_filtered - mean_states.unsqueeze(0), dim=2),  # type: ignore[reportUnknownMemberType]
            )

            # Mean radius (B,)
            mean_radius: torch.Tensor = torch.mean(distances, dim=0).unsqueeze(1)  # (B, 1)
            feature_list.append(mean_radius)

            # Max radius (B,)
            max_radius: torch.Tensor = torch.max(distances, dim=0)[0].unsqueeze(1)  # (B, 1)
            feature_list.append(max_radius)

        # 2. PCA eigenvalues of covariance matrix
        if self.include_pca:
            mean_states_pca: torch.Tensor = torch.mean(y_filtered, dim=0)  # (B, S)
            # Center the data: (N, B, S)
            centered: torch.Tensor = y_filtered - mean_states_pca.unsqueeze(0)

            # For each trajectory, compute covariance matrix and eigenvalues
            batch_size = y_filtered.shape[1]
            eigenvalues_list: list[torch.Tensor] = []

            for b in range(batch_size):
                traj_centered: torch.Tensor = centered[:, b, :]  # (N, S)
                # Covariance matrix: (S, S)
                cov_matrix: torch.Tensor = torch.cov(traj_centered.T)
                # Compute eigenvalues
                eigenvals: torch.Tensor = cast(
                    torch.Tensor,
                    torch.linalg.eigvalsh(cov_matrix),  # type: ignore[reportUnknownMemberType]
                )  # (S,)
                # Sort in descending order
                eigenvals_sorted: torch.Tensor = torch.sort(eigenvals, descending=True)[0]
                eigenvalues_list.append(eigenvals_sorted)

            pca_features: torch.Tensor = torch.stack(eigenvalues_list)  # (B, S)
            feature_list.append(pca_features)

        # Concatenate all features
        features: torch.Tensor = torch.cat(feature_list, dim=1)  # (B, F_total)

        return features.to(y_filtered.device)

    @property
    def feature_names(self) -> list[str]:
        """Return the list of feature names.

        Returns:
            List of feature names based on which features are enabled.

        Raises:
            RuntimeError: If extract_features has not been called yet.
        """
        if self._num_states is None:
            raise RuntimeError("Number of states not initialized. Call extract_features first.")

        names: list[str] = []

        if self.include_radius:
            names.extend(["mean_radius", "max_radius"])

        if self.include_pca:
            names.extend([f"pca_eigenvalue_{i}" for i in range(self._num_states)])

        return names

    def _get_num_states(self) -> int:
        """Get the number of state dimensions from the last extraction.

        This is a helper for feature_names. In practice, the number of PCA
        eigenvalues equals the number of state dimensions.
        """
        return 2
