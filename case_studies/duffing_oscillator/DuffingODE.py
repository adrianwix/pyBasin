from typing import TypedDict

import torch
from pybasin.ODESystem import ODESystem


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

    def get_str(self) -> str:
        delta = self.params["delta"]
        k3 = self.params["k3"]
        A = self.params["A"]
        description = (
            f"  dx/dt       = x_dot\n"
            f"  dx_dot/dt   = -({delta})*x_dot - ({k3})*x^3 + ({A})*cos(t)\n"
        )
        return description
