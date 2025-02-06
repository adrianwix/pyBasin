"""
-- ODESystem(ABC): contains the ODE formulation, and optional model parameters
+ method .ode  -> first order ODE definition
+ parameters will be handled over as dictionary
+ method .symbolic_ode [optional] for graphical display (tbd)
--
"""


from abc import ABC, abstractmethod
from typing import TypedDict, List, Dict, TypeVar, Generic
import numpy as np
from numpy.typing import NDArray

P = TypeVar('P', bound=Dict[str, float])


class ODESystem(ABC, Generic[P]):
    """
    Abstract base class for defining an ODE system.
    """

    def __init__(self, params: P):
        self.params = params

    @abstractmethod
    def ode(self, t: float, y: NDArray[np.float64]) -> List[float]:
        """
        Right-hand side (RHS) for the ODE.

        Parameters
        ----------
        t : float
            The current time (not explicitly used if the system is time-invariant).
        y : NDArray[np.float64], shape (n,)
            The current state.

        Returns
        -------
        dydt : List[float]
            The derivatives.
        """
        pass

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

    def ode(self, t: float, y: NDArray[np.float64]) -> List[float]:
        """
        Right-hand side (RHS) for the pendulum ODE.

        Parameters
        ----------
        t : float
            The current time (not explicitly used if the system is time-invariant).
        y : NDArray[np.float64], shape (2,)
            The current state, [theta, theta_dot].

        Returns
        -------
        dydt : List[float]
            The derivatives: [dtheta/dt, dtheta_dot/dt].
        """
        alpha = self.params["alpha"]
        T = self.params["T"]
        K = self.params["K"]

        theta, theta_dot = y
        dtheta_dt = theta_dot
        dtheta_dot_dt = -alpha * theta_dot + T - K * np.sin(theta)

        return [dtheta_dt, dtheta_dot_dt]

    def symbolic_ode(self) -> None:
        """
        Optional method to display the symbolic form of the ODE.
        """
        print("dtheta/dt = theta_dot")
        print("dtheta_dot/dt = -alpha * theta_dot + T - K * sin(theta)")
