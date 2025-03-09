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

    @abstractmethod
    def get_str(self) -> str:
        """
        Returns a string representation of the ODE system with its parameters.

        The string is constructed using multiple line f-string interpolation.
        """
        raise NotImplementedError

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
