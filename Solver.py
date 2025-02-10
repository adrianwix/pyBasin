from abc import ABC, abstractmethod
import numpy as np
import torch
from torchdiffeq import odeint


class Solver(ABC):
    """Abstract base class for ODE solvers."""

    def __init__(self, time_span: tuple[float, float], fs: float, **kwargs):
        """
        Initialize the solver with integration parameters.

        :param time_span: Tuple (t_start, t_end) defining integration interval.
        :param fs: Sampling frequency (Hz) - number of samples per time unit.
        """
        self.time_span = time_span
        self.fs = fs
        dt = 1/fs
        self.n_steps = int((time_span[1] - time_span[0]) * fs) + 1
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
    Expects y0 to be a torch.Tensor and uses sampling frequency for time steps.
    """

    def __init__(self, time_span, fs, **kwargs):
        super().__init__(time_span, fs, **kwargs)

    def integrate(self, ode_system, y0) -> tuple[torch.Tensor, torch.Tensor]:
        # y0 is already a torch.Tensor
        t_start, t_end = self.time_span
        t_eval = torch.linspace(t_start, t_end, self.n_steps)

        y_torch = odeint(ode_system, y0, t_eval, rtol=1e-8)

        return t_eval, y_torch
