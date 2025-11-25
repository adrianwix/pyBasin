"""Protocol definitions for ODE systems and solvers.

This module defines Protocol classes that provide structural typing for
the common interfaces shared by different implementations (e.g., ODESystem
and JaxODESystem, Solver and JaxSolver).

Using Protocol allows type checkers to accept any class that implements
the required methods, without requiring explicit inheritance.
"""

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class ODESystemProtocol(Protocol):
    """Protocol defining the common interface for ODE systems.

    Implementations: ODESystem (PyTorch-based), JaxODESystem (JAX-based).

    Both implementations satisfy this protocol via structural typing (no explicit inheritance needed).
    This allows generic code to work with either implementation.

    Attributes
    ----------
    params : Any
        Parameter dictionary for the ODE system.
    """

    params: Any

    def get_str(self) -> str:
        """
        Returns a string representation of the ODE system with its parameters.

        Used for caching and logging purposes.

        Returns
        -------
        str
            A human-readable description of the ODE system and its parameters.
        """
        ...


@runtime_checkable
class SolverProtocol(Protocol):
    """Protocol defining the common interface for ODE solvers.

    Implementations: Solver (PyTorch-based), JaxSolver (JAX-based).

    Both implementations satisfy this protocol via structural typing (no explicit inheritance needed).
    This allows generic code to work with either implementation.

    Attributes
    ----------
    time_span : tuple[float, float]
        The integration time interval (t_start, t_end).
    n_steps : int
        Number of evaluation points.
    device : torch.device
        Device for output tensors.
    use_cache : bool
        Whether caching is enabled.
    """

    time_span: tuple[float, float]
    n_steps: int
    device: torch.device
    use_cache: bool

    def integrate(self, ode_system: Any, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the ODE system and return the evaluation time points and solution.

        Parameters
        ----------
        ode_system : ODESystemProtocol
            An instance of an ODE system (ODESystem or JaxODESystem).
        y0 : torch.Tensor
            Initial conditions with shape (batch, n_dims).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple (t_eval, y_values) where y_values has shape (n_steps, batch, n_dims).
        """
        ...

    def with_device(self, device: str) -> "SolverProtocol":
        """
        Create a copy of this solver configured for a different device.

        Parameters
        ----------
        device : str
            Target device ('cpu', 'cuda', 'gpu').

        Returns
        -------
        SolverProtocol
            New solver instance with the same configuration but different device.
        """
        ...
