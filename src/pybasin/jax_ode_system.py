# pyright: reportUntypedBaseClass=false
"""JAX-native ODE system base class.

This module provides a base class for defining ODE systems using pure JAX operations,
enabling JIT compilation and efficient GPU execution without PyTorch callbacks.
"""

from abc import ABC, abstractmethod
from typing import TypeVar

from jax import Array

# TypeVar for parameter dictionaries
P = TypeVar("P")


class JaxODESystem[P](ABC):
    """
    Abstract base class for defining an ODE system using pure JAX.

    This class is designed for ODE systems that need maximum performance with JAX/Diffrax.
    Unlike the PyTorch-based ODESystem, this uses pure JAX operations that can be
    JIT-compiled for optimal GPU performance.

    Type Parameters
    ---------------
    P : dict or TypedDict
        The parameter dictionary type for this ODE system.
        Should be a TypedDict subclass for best type checking.

    Examples
    --------
    >>> from typing import TypedDict
    >>> import jax.numpy as jnp
    >>> from jax import Array
    >>>
    >>> class MyParams(TypedDict):
    ...     alpha: float
    ...     beta: float
    >>>
    >>> class MyJaxODE(JaxODESystem[MyParams]):
    ...     def __init__(self, params: MyParams):
    ...         super().__init__(params)
    ...
    ...     def ode(self, t: Array, y: Array) -> Array:
    ...         alpha = self.params["alpha"]
    ...         return jnp.zeros_like(y)
    ...
    ...     def get_str(self) -> str:
    ...         return f"MyODE with alpha={self.params['alpha']}"
    """

    def __init__(self, params: P) -> None:
        """
        Initialize the JAX ODE system.

        Parameters
        ----------
        params : P
            Dictionary of ODE parameters.
        """
        self.params = params

    @abstractmethod
    def ode(self, t: Array, y: Array) -> Array:
        """
        Right-hand side (RHS) for the ODE using pure JAX operations.

        This method must use only JAX operations (jnp, not np or torch)
        to enable JIT compilation and efficient execution.

        Parameters
        ----------
        t : Array
            The current time (scalar JAX array).
        y : Array
            The current state with shape (n_dims,) for single trajectory.

        Returns
        -------
        derivatives : Array
            The time derivatives with the same shape as y.

        Notes
        -----
        - Use jnp operations instead of np or torch
        - Avoid Python control flow that depends on array values
        - This method will be JIT-compiled, so ensure it's traceable
        """
        pass

    @abstractmethod
    def get_str(self) -> str:
        """
        Returns a string representation of the ODE system with its parameters.

        Used for caching and logging purposes.

        Returns
        -------
        str
            A human-readable description of the ODE system and its parameters.
        """
        pass

    def __call__(self, t: Array, y: Array, args: None = None) -> Array:
        """
        Make the ODE system callable for use with Diffrax.

        Diffrax expects f(t, y, args) signature.

        Parameters
        ----------
        t : Array
            Current time.
        y : Array
            Current state.
        args : None
            Unused, present for Diffrax compatibility.

        Returns
        -------
        Array
            Time derivatives.
        """
        return self.ode(t, y)
