from typing import TypedDict

import torch

from pybasin.ode_system import ODESystem


class PendulumParams(TypedDict):
    """Parameters for the pendulum ODE system."""

    alpha: float  # damping coefficient
    T: float  # external torque
    K: float  # stiffness coefficient


class PendulumODE(ODESystem[PendulumParams]):
    def __init__(self, params: PendulumParams):
        super().__init__(params)

    # TODO: Remove t from the signature if not used
    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Vectorized right-hand side (RHS) for the pendulum ODE using PyTorch.
        """
        alpha = self.params["alpha"]
        torque = self.params["T"]
        k = self.params["K"]

        theta = y[..., 0]
        theta_dot = y[..., 1]

        dtheta_dt = theta_dot
        dtheta_dot_dt = -alpha * theta_dot + torque - k * torch.sin(theta)

        return torch.stack([dtheta_dt, dtheta_dot_dt], dim=1)

    def get_str(self) -> str:
        """
        Returns a string representation of the ODE system with its parameters.

        The string is constructed using multiple line f-string interpolation.
        """
        alpha = self.params["alpha"]
        torque = self.params["T"]
        k = self.params["K"]
        description = (
            f"  dtheta/dt      = theta_dot\n"
            f"  dtheta_dot/dt  = -({alpha}) * theta_dot + ({torque}) - ({k}) * sin(theta)\n"
        )
        return description
