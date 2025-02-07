from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp

class Solver(ABC):
    """Abstract base class for ODE solvers."""

    def __init__(self, time_span, method, rtol=1e-8, **kwargs):
        """
        Initialize the solver with integration parameters.

        :param time_span: Tuple (t_start, t_end) defining integration interval.
        :param method: Integration method for solve_ivp (e.g., 'RK45', 'RK23', 'BDF').
        :param rtol: Relative tolerance for integration.
        """
        self.time_span = time_span
        self.method = method
        self.rtol = rtol
        self.params = kwargs  # Additional solver parameters

    @abstractmethod
    def integrate(self, ode_system, y0):
        """
        Abstract method to solve an ODE system.
        
        :param ode_system: Function representing the ODE system.
        :param y0: Initial conditions.
        :return: (t_values, y_values) where y_values is shape (len(t_values), N).
        """
        pass


class SciPySolver(Solver):
    """ODE solver using SciPy's solve_ivp."""

    def integrate(self, ode_system, y0):
        sol = solve_ivp(
            fun=ode_system,
            t_span=self.time_span,
            y0=y0,
            method=self.method,
            rtol=self.rtol,
            dense_output=True
        )
        return sol.t, sol.y.T  # Transposing for consistency (shape => (len(t), N))
