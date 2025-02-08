from abc import ABC, abstractmethod
from typing import TypedDict, List, Dict, TypeVar, Generic
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn

P = TypeVar('P', bound=Dict[str, float])


class ODESystem(ABC, Generic[P], nn.Module):
    """
    Abstract base class for defining an ODE system.
    """

    def __init__(self, params: P):
        super(ODESystem, self).__init__()  # Initialize nn.Module
        self.params = params

    @abstractmethod
    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Right-hand side (RHS) for the ODE using PyTorch tensors.

        Parameters
        ----------
        t : torch.Tensor
            The current time (can be scalar or batch).
        y : torch.Tensor
            The current state (can be shape (..., n)).

        Returns
        -------
        derivatives : torch.Tensor
            The time derivatives with the same leading shape as y.
        """
        pass

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calls the ODE function in a manner consistent with nn.Module.
        """
        return self.ode(t, y)

    def symbolic_ode(self) -> None:
        """
        Optional method to display the symbolic form of the ODE.
        """
        pass


class PendulumParams(TypedDict):
    alpha: float
    T: float
    K: float


class PendulumODE(ODESystem[PendulumParams]):
    def __init__(self, params: PendulumParams):
        super().__init__(params)

    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Vectorized right-hand side (RHS) for the pendulum ODE using PyTorch.
        """
        alpha = self.params["alpha"]
        T = self.params["T"]
        K = self.params["K"]

        theta = y[..., 0]
        theta_dot = y[..., 1]

        dtheta_dt = theta_dot
        dtheta_dot_dt = -alpha * theta_dot + T - K * torch.sin(theta)

        return torch.stack([dtheta_dt, dtheta_dot_dt], dim=1)

    def symbolic_ode(self) -> None:
        """
        Optional method to display the symbolic form of the ODE.
        """
        print("dtheta/dt = theta_dot")
        print("dtheta_dot/dt = -alpha * theta_dot + T - K * sin(theta)")


class DuffingParams(TypedDict):
    delta: float  # Damping coefficient
    k3: float    # Cubic stiffness
    A: float     # Forcing amplitude


class DuffingODE(ODESystem[DuffingParams]):
    """
    Duffing oscillator ODE system.
    Following Thomson & Steward: Nonlinear Dynamics and Chaos. Page 9, Fig. 1.9.

    For 5 multistability, recommended parameters are:
    delta = 0.08, k3 = 1, A = 0.2
    """

    def __init__(self, params: DuffingParams):
        super().__init__(params)

    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Vectorized right-hand side (RHS) for the Duffing oscillator ODE using PyTorch.
        """
        delta = self.params["delta"]
        k3 = self.params["k3"]
        A = self.params["A"]

        x = y[..., 0]
        x_dot = y[..., 1]

        dx_dt = x_dot
        dx_dot_dt = -delta * x_dot - k3 * x**3 + A * torch.cos(t)

        return torch.stack([dx_dt, dx_dot_dt], dim=1)

    def symbolic_ode(self) -> None:
        """
        Display the symbolic form of the Duffing oscillator ODE.
        """
        print("dx/dt = x_dot")
        print("dx_dot/dt = -delta * x_dot - k3 * x^3 + A * cos(t)")
