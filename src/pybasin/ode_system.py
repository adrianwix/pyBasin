from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
import torch.nn as nn

# TypeVar for parameter dictionaries
# Using bound=dict to allow both dict and TypedDict instances
P = TypeVar("P")


class ODESystem(ABC, Generic[P], nn.Module):
    """
    Abstract base class for defining an ODE system.

    Type Parameters
    ---------------
    P : dict or TypedDict
        The parameter dictionary type for this ODE system.
        Should be a TypedDict subclass for best type checking.

    Examples
    --------
    >>> from typing import TypedDict
    >>> class MyParams(TypedDict):
    ...     alpha: float
    ...     beta: float
    >>>
    >>> class MyODE(ODESystem[MyParams]):
    ...     def __init__(self, params: MyParams):
    ...         super().__init__(params)
    ...
    ...     def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ...         # self.params is typed as MyParams
    ...         alpha = self.params["alpha"]  # type checker knows this exists
    ...         return torch.zeros_like(y)
    """

    def __init__(self, params: P) -> None:
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
