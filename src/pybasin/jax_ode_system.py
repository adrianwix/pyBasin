# pyright: reportUntypedBaseClass=false
"""JAX-native ODE system base class.

This module provides a base class for defining ODE systems using pure JAX operations,
enabling JIT compilation and efficient GPU execution without PyTorch callbacks.
"""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from jax import Array

# TypeVar for parameter dictionaries
P = TypeVar("P")


class JaxODESystem[P](ABC):
    """
    Abstract base class for defining an ODE system using pure JAX.

    This class is designed for ODE systems that need maximum performance with JAX/Diffrax.
    Unlike the PyTorch-based ODESystem, this uses pure JAX operations that can be
    JIT-compiled for optimal GPU performance.

    ```python
    from typing import TypedDict
    import jax.numpy as jnp
    from jax import Array


    class MyParams(TypedDict):
        alpha: float
        beta: float


    class MyJaxODE(JaxODESystem[MyParams]):
        def __init__(self, params: MyParams):
            super().__init__(params)

        def ode(self, t: Array, y: Array) -> Array:
            alpha = self.params["alpha"]
            return jnp.zeros_like(y)

        def get_str(self) -> str:
            return f"MyODE with alpha={self.params['alpha']}"
    ```

    :param P: Type parameter - the parameter dictionary type for this ODE system.
        Should be a TypedDict subclass for best type checking.
    """

    def __init__(self, params: P) -> None:
        """
        Initialize the JAX ODE system.

        :param params: Dictionary of ODE parameters.
        """
        self.params = params

    @abstractmethod
    def ode(self, t: Array, y: Array) -> Array:
        """
        Right-hand side (RHS) for the ODE using pure JAX operations.

        This method must use only JAX operations (jnp, not np or torch)
        to enable JIT compilation and efficient execution.

        Notes:

        - Use jnp operations instead of np or torch
        - Avoid Python control flow that depends on array values
        - This method will be JIT-compiled, so ensure it's traceable

        :param t: The current time (scalar JAX array).
        :param y: The current state with shape (n_dims,) for single trajectory.
        :return: The time derivatives with the same shape as y.
        """
        pass

    @abstractmethod
    def get_str(self) -> str:
        """
        Returns a string representation of the ODE system with its parameters.

        Used for caching and logging purposes.

        :return: A human-readable description of the ODE system and its parameters.
        """
        pass

    def to(self, device: Any) -> "JaxODESystem[P]":
        """No-op for JAX systems - device handling is done on tensors.

        This method exists for API compatibility with PyTorch-based ODESystem.

        :param device: Ignored for JAX systems.
        :return: Returns self.
        """
        return self

    def __call__(self, t: Array, y: Array, args: None = None) -> Array:
        """
        Make the ODE system callable for use with Diffrax.

        Diffrax expects f(t, y, args) signature.

        :param t: Current time.
        :param y: Current state.
        :param args: Unused, present for Diffrax compatibility.
        :return: Time derivatives.
        """
        return self.ode(t, y)
