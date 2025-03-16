import numpy as np
import torch
from typing import Optional, Dict, Any


class Solution:
    """
    Solution: Represents the time integration result for a single initial condition.

    This class stores:
        - The initial condition used for integration.
        - The time series result from integration.
        - Features extracted from the trajectory.
        - An optional label/classification.
        - Optional model parameters that were used in the integration.

    Attributes:
        initial_condition (torch.Tensor): The initial condition used for integration.
        time (np.ndarray): Time points of the solution.
        y (torch.Tensor): State values over time (shape: (len(time), len(initial_conditions), num_states)).
        features (Optional[np.ndarray]): Extracted features (e.g., steady-state properties).
        label (Optional[Any]): A label assigned to the solution (e.g., classification result).
        model_params (Optional[Dict[str, Any]]): Parameters of the ODE model.
    """

    def __init__(
        self,
        initial_condition: torch.Tensor,
        time: torch.Tensor,
        y: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        labels: Optional[np.ndarray] = None,
        model_params: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the Solution object.

        :param initial_condition: shape: (B, S) => B batches, S state variables
        :param time: shape: (N,) => N time points
        :param y: shape: (N, B, S) => N time points, B batches, S state variables
        :param features: Optional features describing the trajectory.
        :param label: Optional classification label for the solution.
        :param model_params: Optional dictionary of model parameters used in the simulation.
        """
        # Assertions for shape checks
        assert initial_condition.ndim == 2, "initial_condition must be 2D (B, S)"
        assert time.ndim == 1, "time must be 1D (N,)"
        assert y.ndim == 3, "y must be 3D (N, B, S)"
        assert y.shape[0] == time.shape[0], (
            f"Time dimension mismatch: y.shape[0] ({y.shape[0]}) != len(time) ({time.shape[0]})"
        )
        assert y.shape[1] == initial_condition.shape[0], (
            f"Initial conditions dimension mismatch: y.shape[1] ({y.shape[1]}) != initial_condition.shape[0] ({initial_condition.shape[0]})"
        )
        assert y.shape[2] == initial_condition.shape[1], (
            f"State dimension mismatch: y.shape[2] ({y.shape[1]}) != initial_condition.shape[1] ({initial_condition.shape[0]})"
        )

        self.initial_condition = initial_condition
        self.time = time
        self.y = y  # shape: (N, B, S)
        self.features = features if features is not None else None
        self.labels = labels
        self.model_params = model_params
        self.bifurcation_amplitudes = None

    def set_labels(self, labels: np.ndarray):
        """
        Assign a label to this solution.

        :param labels: The label to assign from the classification results.
        """
        assert len(labels) == self.y.shape[1], (
            f"Initial conditions dimension mismatch: len(labels) ({len(labels)}) != y.shape[1] ({self.y.shape[1]})"
        )

        self.labels = labels

    def set_features(self, features: torch.Tensor):
        """
        Store features extracted from the trajectory.

        :param features: A feature vector describing the solution.
        """
        self.features = features

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a summary of the solution.

        :return: Dictionary with key information about the solution.
        """
        return {
            "initial_condition": self.initial_condition.tolist(),
            "num_time_steps": len(self.time),
            "trajectory_shape": self.y.shape,  # Updated to use self.y
            "features": self.features.tolist() if self.features is not None else None,
            "label": self.label,
            "model_params": self.model_params,
        }

    def __repr__(self) -> str:
        return f"Solution(IC={self.initial_condition}, Label={self.label}, Features={self.features})"
