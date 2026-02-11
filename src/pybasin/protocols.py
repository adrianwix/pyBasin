"""Protocol definitions for ODE systems and solvers.

This module defines Protocol classes that provide structural typing for
the common interfaces shared by different implementations (e.g., ODESystem
and JaxODESystem, Solver and JaxSolver).

Using Protocol allows type checkers to accept any class that implements
the required methods, without requiring explicit inheritance.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class SklearnClassifier(Protocol):
    """Protocol for sklearn-compatible classifiers.

    Any class with ``fit()`` and ``predict()`` methods satisfies this protocol.
    Used for type narrowing in BSE when ``is_classifier()`` returns True.
    """

    def fit(self, X: np.ndarray, y: Any) -> Any: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class SklearnClusterer(Protocol):
    """Protocol for sklearn-compatible clusterers.

    Any class with ``fit_predict()`` satisfies this protocol.
    Used for type narrowing in BSE when ``is_clusterer()`` returns True.
    """

    def fit_predict(self, X: np.ndarray, y: Any = None) -> np.ndarray: ...


@runtime_checkable
class FeatureNameAware(Protocol):
    """Protocol for predictors that accept feature names.

    Predictors like ``HDBSCANClusterer`` and ``DynamicalSystemClusterer``
    use feature names for domain-specific logic.
    """

    def set_feature_names(self, feature_names: list[str]) -> None: ...


@runtime_checkable
class ODESystemProtocol(Protocol):
    """Protocol defining the common interface for ODE systems.

    Implementations: ODESystem (PyTorch-based), JaxODESystem (JAX-based).

    Both implementations satisfy this protocol via structural typing (no explicit inheritance needed).
    This allows generic code to work with either implementation.

    :ivar params: Parameter dictionary for the ODE system.
    :vartype params: Any
    """

    params: Any

    def to(self, device: Any) -> "ODESystemProtocol":
        """Move the ODE system to the specified device.

        For PyTorch-based systems, this moves the module to the device.
        For JAX systems, this is a no-op (returns self).

        :param device: The target device.
        :return: The ODE system on the target device.
        """
        ...

    def get_str(self) -> str:
        """
        Returns a string representation of the ODE system with its parameters.

        Used for caching and logging purposes.

        :return: A human-readable description of the ODE system and its parameters.
        """
        ...


@runtime_checkable
class SolverProtocol(Protocol):
    """Protocol defining the common interface for ODE solvers.

    Implementations: Solver (PyTorch-based), JaxSolver (JAX-based).

    Both implementations satisfy this protocol via structural typing (no explicit inheritance needed).
    This allows generic code to work with either implementation.

    :ivar time_span: The integration time interval (t_start, t_end).
    :ivar n_steps: Number of evaluation points.
    :ivar device: Device for output tensors.
    :ivar use_cache: Whether caching is enabled.

    :vartype time_span: tuple[float, float]
    :vartype n_steps: int
    :vartype device: torch.device
    :vartype use_cache: bool
    """

    time_span: tuple[float, float]
    n_steps: int
    device: torch.device
    use_cache: bool

    def integrate(
        self, ode_system: ODESystemProtocol, y0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the ODE system and return the evaluation time points and solution.

        :param ode_system: An instance of an ODE system (ODESystem or JaxODESystem).
        :param y0: Initial conditions with shape (batch, n_dims).
        :return: Tuple (t_eval, y_values) where y_values has shape (n_steps, batch, n_dims).
        """
        ...

    def with_device(self, device: str) -> "SolverProtocol":
        """
        Create a copy of this solver configured for a different device.

        :param device: Target device ('cpu', 'cuda', 'gpu').
        :return: New solver instance with the same configuration but different device.
        """
        ...
