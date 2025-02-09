from abc import ABC, abstractmethod
import numpy as np
import torch
from torchdiffeq import odeint


class Solver(ABC):
    """Abstract base class for ODE solvers."""

    def __init__(self, time_span: tuple[int, int], N: int, **kwargs):
        """
        Initialize the solver with integration parameters.

        :param time_span: Tuple (t_start, t_end) defining integration interval.
        :param rtol: Relative tolerance for integration.
        """
        self.time_span = time_span
        self.n_steps = N
        self.params = kwargs  # Additional solver parameters

    @abstractmethod
    def integrate(self, ode_system, y0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Abstract method to solve an ODE system.

        :param ode_system: Function representing the ODE system.
        :param y0: Initial conditions.
        :return: (t_values, y_values) where y_values is shape (len(t_values), N).
        """
        pass


class TorchDiffEqSolver(Solver):
    """
    ODE solver using torchdiffeq. 
    Expects y0 to be a torch.Tensor and uses N as the total number of steps.
    """

    def __init__(self, time_span, N, **kwargs):
        super().__init__(time_span, N, **kwargs)

    def integrate(self, ode_system, y0) -> tuple[torch.Tensor, torch.Tensor]:
        # y0 is already a torch.Tensor
        t_start, t_end = self.time_span
        t_eval = torch.linspace(t_start, t_end, self.n_steps)

        y_torch = odeint(ode_system, y0, t_eval)

        return t_eval, y_torch
