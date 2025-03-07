from abc import ABC, abstractmethod
from typing import TypedDict, Dict, TypeVar, Generic
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


class LorenzParams(TypedDict):
    sigma: float  # Prandtl number
    r: float     # Rayleigh number
    b: float     # Physical dimension parameter


class LorenzODE(ODESystem[LorenzParams]):
    """
    Lorenz system ODE.

    Classical parameter choice:
        sigma = 10, r = 28, b = 8/3

    For broken butterfly (https://doi.org/10.1142/S0218127414501314):
        sigma = 0.12, r = 0, b = -0.6
    """

    def __init__(self, params: LorenzParams):
        super().__init__(params)

    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Vectorized right-hand side (RHS) for the Lorenz system using PyTorch,
        modified so that once a sample’s state magnitude exceeds 200, its derivative is set to zero.

        y shape: (..., 3) where the last dimension represents [x, y, z]
        returns: tensor with the same shape as y.
        """
        # Extract parameters
        sigma = self.params["sigma"]
        r = self.params["r"]
        b = self.params["b"]

        # Unpack state variables
        x = y[..., 0]
        y_ = y[..., 1]  # avoid shadowing y parameter
        z = y[..., 2]

        # Compute standard Lorenz dynamics
        dx_dt = sigma * (y_ - x)
        dy_dt = r * x - x * z - y_
        dz_dt = x * y_ - b * z

        # Usually we woudl return here, but we want to modify the dynamics
        # based on the state of the system. To handle out-of-bounds states

        # Stack into the derivative tensor
        dydt = torch.stack([dx_dt, dy_dt, dz_dt], dim=-1)

        # TODO: Can we move this masking logic into a method?
        # Create a mask: for each sample, check if the maximum absolute value is less than 200.
        # If yes, mask=1 (continue dynamics); if not, mask=0 (freeze dynamics).
        mask = (torch.max(torch.abs(y), dim=-1)[0] < 200).float().unsqueeze(-1)

        # Return the dynamics modified by the mask so that terminated samples evolve with zero derivative.
        return dydt * mask

    def symbolic_ode(self) -> None:
        """Display the symbolic form of the Lorenz system."""
        print("dx/dt = sigma * (y - x)")
        print("dy/dt = r*x - x*z - y")
        print("dz/dt = x*y - b*z")
