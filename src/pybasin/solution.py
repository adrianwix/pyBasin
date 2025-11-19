from typing import Any

import numpy as np
import torch


class Solution:
    """
    Solution: Represents the time integration result for a single initial condition.

    This class stores:
        - The initial condition used for integration.
        - The time series result from integration.
        - Features extracted from the trajectory.
        - Optional labels/classification for each trajectory.
        - Optional model parameters that were used in the integration.
        - Optional bifurcation amplitudes extracted from the trajectory.

    Attributes:
        initial_condition (torch.Tensor): The initial condition used for integration (shape: B, S).
        time (torch.Tensor): Time points of the solution (shape: N).
        y (torch.Tensor): State values over time (shape: N, B, S).
        features (torch.Tensor | None): Extracted features (e.g., steady-state properties).
        labels (np.ndarray | None): Labels assigned to each solution in the batch.
        model_params (dict[str, float] | None): Parameters of the ODE model.
        bifurcation_amplitudes (torch.Tensor | None): Maximum absolute values along time dimension.
    """

    def __init__(
        self,
        initial_condition: torch.Tensor,
        time: torch.Tensor,
        y: torch.Tensor,
        features: torch.Tensor | None = None,
        labels: np.ndarray | None = None,
        model_params: dict[str, float] | None = None,
    ):
        """
        Initialize the Solution object.

        :param initial_condition: shape: (B, S) => B batches, S state variables
        :param time: shape: (N,) => N time points
        :param y: shape: (N, B, S) => N time points, B batches, S state variables
        :param features: Optional features describing the trajectory.
        :param labels: Optional classification labels for the solutions.
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
        self.bifurcation_amplitudes: torch.Tensor | None = None

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

    def get_summary(self) -> dict[str, Any]:
        """
        Return a summary of the solution.

        :return: Dictionary with key information about the solution.
        """
        initial_condition_list: list[Any] = self.initial_condition.tolist()  # type: ignore[assignment]
        features_list: list[Any] | None = None
        if self.features is not None:
            features_list = self.features.tolist()  # type: ignore[assignment]
        labels_list: list[Any] | None = self.labels.tolist() if self.labels is not None else None

        return {
            "initial_condition": initial_condition_list,
            "num_time_steps": len(self.time),
            "trajectory_shape": self.y.shape,
            "features": features_list,
            "labels": labels_list,
            "model_params": self.model_params,
        }
