from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
import torch.nn as nn

P = TypeVar("P", bound=dict[str, float])


class ODESystem(ABC, Generic[P], nn.Module):
    """
    Abstract base class for defining an ODE system.
    """

    def __init__(self, params: P):
        # Initialize nn.Module
        super().__init__()  # type: ignore
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
