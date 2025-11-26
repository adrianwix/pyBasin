"""Feature extractors using nolds library for nonlinear dynamics analysis."""

import nolds  # pyright: ignore[reportMissingTypeStubs]
import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution


class LyapunovFeatureExtractor(FeatureExtractor):
    """Extract Lyapunov exponent features using nolds.

    Computes the largest Lyapunov exponent for each trajectory, which indicates
    whether the attractor is:
    - Positive: Chaotic (sensitive dependence on initial conditions)
    - Near zero: Regular periodic (limit cycle, torus)
    - Negative: Fixed point

    Args:
        time_steady: Time threshold for filtering transients. Default 0.0.
        emb_dim: Embedding dimension for Lyapunov calculation. Default 10.
        matrix_dim: Matrix dimension for Lyapunov calculation. Default 4.
    """

    def __init__(
        self,
        time_steady: float = 0.0,
        emb_dim: int = 10,
        matrix_dim: int = 4,
    ):
        super().__init__(time_steady=time_steady)
        self.emb_dim = emb_dim
        self.matrix_dim = matrix_dim

    def extract_features(self, solution: Solution) -> torch.Tensor:
        """Extract Lyapunov exponent features.

        Args:
            solution: ODE solution with shape (N, B, S)

        Returns:
            Features tensor of shape (B, S) containing Lyapunov exponents
            for each state variable of each trajectory.
        """

        # Filter time
        y_filtered = self.filter_time(solution)

        # y_filtered shape: (N, B, S)
        _n, batch_size, num_states = y_filtered.shape

        # Convert to numpy for nolds
        y_np = y_filtered.cpu().numpy()

        # Compute Lyapunov exponent for each trajectory and state
        lyapunov_features: list[list[float]] = []

        for b in range(batch_size):
            traj_features: list[float] = []
            for s in range(num_states):
                time_series = y_np[:, b, s]
                try:
                    # Compute largest Lyapunov exponent
                    lyap = nolds.lyap_r(  # type: ignore[misc]
                        time_series,
                        emb_dim=self.emb_dim,
                        trajectory_len=self.matrix_dim,
                    )
                    traj_features.append(lyap)  # type: ignore[arg-type]
                except Exception:
                    # If calculation fails, use NaN (will be imputed later)
                    traj_features.append(float("nan"))

            lyapunov_features.append(traj_features)

        # Convert to tensor: (B, S)
        features = torch.tensor(lyapunov_features, dtype=torch.float32)

        return features


class CorrelationDimensionExtractor(FeatureExtractor):
    """Extract correlation dimension features using nolds.

    Computes the correlation dimension for each trajectory, which indicates
    the fractal dimension of the attractor:
    - ~0: Point attractor (fixed point)
    - ~1: Simple limit cycle
    - ~2: Torus
    - Non-integer/higher: Strange attractor (chaos)

    Args:
        time_steady: Time threshold for filtering transients. Default 0.0.
        emb_dim: Embedding dimension for correlation dimension calculation. Default 10.
    """

    def __init__(
        self,
        time_steady: float = 0.0,
        emb_dim: int = 10,
    ):
        super().__init__(time_steady=time_steady)
        self.emb_dim = emb_dim

    def extract_features(self, solution: Solution) -> torch.Tensor:
        """Extract correlation dimension features.

        Args:
            solution: ODE solution with shape (N, B, S)

        Returns:
            Features tensor of shape (B, S) containing correlation dimensions
            for each state variable of each trajectory.
        """

        # Filter time
        y_filtered = self.filter_time(solution)

        # y_filtered shape: (N, B, S)
        _n, batch_size, num_states = y_filtered.shape

        # Convert to numpy for nolds
        y_np = y_filtered.cpu().numpy()

        # Compute correlation dimension for each trajectory and state
        corr_dim_features: list[list[float]] = []

        for b in range(batch_size):
            traj_features: list[float] = []
            for s in range(num_states):
                time_series = y_np[:, b, s]
                try:
                    # Compute correlation dimension
                    corr_dim = nolds.corr_dim(time_series, emb_dim=self.emb_dim)  # type: ignore[misc]
                    traj_features.append(corr_dim)  # type: ignore[arg-type]
                except Exception:
                    # If calculation fails, use NaN (will be imputed later)
                    traj_features.append(float("nan"))

            corr_dim_features.append(traj_features)

        # Convert to tensor: (B, S)
        features = torch.tensor(corr_dim_features, dtype=torch.float32)

        return features
