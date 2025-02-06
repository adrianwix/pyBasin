import numpy as np
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
        initial_condition (np.ndarray): The initial condition used for integration.
        time (np.ndarray): Time points of the solution.
        trajectory (np.ndarray): State values over time (shape: (len(time), num_states)).
        features (Optional[np.ndarray]): Extracted features (e.g., steady-state properties).
        label (Optional[Any]): A label assigned to the solution (e.g., classification result).
        model_params (Optional[Dict[str, Any]]): Parameters of the ODE model.
    """

    def __init__(
        self,
        initial_condition: np.ndarray,
        time: np.ndarray,
        trajectory: np.ndarray,
        features: Optional[np.ndarray] = None,
        label: Optional[Any] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Solution object.

        :param initial_condition: The initial condition used for integration (shape: (num_states,)).
        :param time: The time values of the integration.
        :param trajectory: The state trajectory over time (shape: (len(time), num_states)).
        :param features: Optional features describing the trajectory.
        :param label: Optional classification label for the solution.
        :param model_params: Optional dictionary of model parameters used in the simulation.
        """
        self.initial_condition = np.array(initial_condition)
        self.time = np.array(time)
        self.trajectory = np.array(trajectory)
        self.features = np.array(features) if features is not None else None
        self.label = label
        self.model_params = model_params

    def assign_label(self, label: Any):
        """
        Assign a label to this solution.

        :param label: The label to assign (e.g., a classification result).
        """
        self.label = label

    def set_features(self, features: np.ndarray):
        """
        Store features extracted from the trajectory.

        :param features: A feature vector describing the solution.
        """
        self.features = np.array(features)

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a summary of the solution.

        :return: Dictionary with key information about the solution.
        """
        return {
            "initial_condition": self.initial_condition.tolist(),
            "num_time_steps": len(self.time),
            "trajectory_shape": self.trajectory.shape,
            "features": self.features.tolist() if self.features is not None else None,
            "label": self.label,
            "model_params": self.model_params,
        }

    def __repr__(self) -> str:
        return f"Solution(IC={self.initial_condition}, Label={self.label}, Features={self.features})"


