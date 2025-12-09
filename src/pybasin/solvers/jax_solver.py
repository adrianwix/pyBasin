# pyright: reportUntypedBaseClass=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
"""Native JAX ODE solver for JaxODESystem.

This module provides a high-performance JAX/Diffrax solver for ODE systems
defined using pure JAX operations. This is the fastest solver option when
using JAX-native ODE systems.
"""

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import torch
from diffrax import (  # type: ignore[import-untyped]
    Dopri5,
    Event,
    ODETerm,
    PIDController,
    SaveAt,
    diffeqsolve,
)
from jax import Array

from pybasin.cache_manager import CacheManager
from pybasin.jax_ode_system import JaxODESystem
from pybasin.jax_utils import get_jax_device, jax_to_torch, torch_to_jax
from pybasin.protocols import ODESystemProtocol
from pybasin.utils import resolve_folder


class JaxSolver:
    """
    High-performance ODE solver using JAX and Diffrax for native JAX ODE systems.

    This solver is optimized for JaxODESystem instances and provides the fastest
    integration performance by avoiding any PyTorch callbacks. It uses JIT
    compilation and vmap for efficient batch processing.

    The interface is compatible with other solvers - it accepts PyTorch tensors
    and returns PyTorch tensors, but internally uses JAX for maximum performance.

    Example
    -------
    >>> from pybasin.jax_ode_system import JaxODESystem
    >>> from pybasin.solvers import JaxSolver
    >>> import torch
    >>>
    >>> class MyODE(JaxODESystem):
    ...     def ode(self, t, y):
    ...         return -y  # Simple decay
    ...     def get_str(self):
    ...         return "decay"
    >>>
    >>> solver = JaxSolver(time_span=(0, 10), n_steps=100)
    >>> y0 = torch.tensor([[1.0, 2.0]])  # batch=1, dims=2
    >>> t, y = solver.integrate(MyODE({}), y0)
    """

    display_name: str = "JAX Solver"

    def __init__(
        self,
        time_span: tuple[float, float],
        n_steps: int | None = None,
        device: str | None = None,
        solver: Any | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_steps: int = 16**5,
        use_cache: bool = True,
        event_fn: Callable[[Any, Array, Any], Array] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize JaxSolver.

        :param time_span: Tuple (t_start, t_end) defining the integration interval.
        :param n_steps: Number of evaluation points. If None, defaults to 500.
        :param device: Device to use ('cuda', 'gpu', 'cpu', or None for auto-detect).
        :param solver: Diffrax solver instance (e.g., Dopri5(), Tsit5()). If None, defaults to Dopri5().
        :param rtol: Relative tolerance for adaptive stepping.
        :param atol: Absolute tolerance for adaptive stepping.
        :param max_steps: Maximum number of steps for the integrator.
        :param use_cache: Whether to use caching for integration results.
        :param event_fn: Optional event function for early termination. Should return positive
                         when integration should continue, negative/zero to stop.
                         Signature: (t, y, args) -> scalar Array.
        """
        self.time_span = time_span
        self.use_cache = use_cache
        self.params = kwargs
        self.event_fn = event_fn

        self._set_n_steps(n_steps)
        self._set_device(device)

        # Diffrax solver settings
        self.diffrax_solver = solver if solver is not None else Dopri5()
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

        # Only create cache manager if caching is enabled
        # This avoids resolve_folder issues when called from threads
        self._cache_manager: CacheManager | None = None
        if use_cache:
            self._cache_manager = CacheManager(resolve_folder("cache"))

    def _set_n_steps(self, n_steps: int | None) -> None:
        """
        Set the number of evaluation steps.

        :param n_steps: Number of evaluation points. If None, defaults to 500.
        """
        self.n_steps = n_steps if n_steps is not None else 500

    def _set_device(self, device: str | None) -> None:
        """
        Set the device for tensor operations with auto-detection.

        :param device: Device to use ('cuda', 'gpu', 'cpu', or None for auto-detect).
        """
        # Store original device string for with_device()
        self._device_str = device

        self.jax_device: Any = get_jax_device(device)

        # PyTorch device for output tensors
        self.device = torch.device("cuda:0" if self.jax_device.platform == "gpu" else "cpu")

    def with_device(self, device: str) -> "JaxSolver":
        """
        Create a copy of this solver configured for a different device.

        :param device: Target device ('cpu', 'cuda', 'gpu').
        :return: New JaxSolver instance with the same configuration but different device.
        """
        return JaxSolver(
            time_span=self.time_span,
            n_steps=self.n_steps,
            device=device,
            solver=self.diffrax_solver,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
            use_cache=self.use_cache,
            event_fn=self.event_fn,
            **self.params,
        )

    def _get_cache_config(self) -> dict[str, Any]:
        """Include solver type, rtol, atol, and max_steps in cache key."""
        return {
            "solver": self.diffrax_solver.__class__.__name__,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_steps": self.max_steps,
        }

    def integrate(
        self, ode_system: ODESystemProtocol, y0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the ODE system and return the evaluation time points and solution.

        :param ode_system: An instance of JaxODESystem.
        :param y0: Initial conditions as PyTorch tensor with shape (batch, n_dims).
        :return: Tuple (t_eval, y_values) as PyTorch tensors where y_values has shape (n_steps, batch, n_dims).
        """
        # Validate y0 shape
        if y0.ndim != 2:
            raise ValueError(
                f"y0 must be 2D with shape (batch, n_dims), got shape {y0.shape}. "
                f"For single initial condition, use y0.unsqueeze(0) or y0.reshape(1, -1)."
            )

        # Convert PyTorch tensor to JAX array
        y0_jax = torch_to_jax(y0, self.jax_device)

        # Prepare time evaluation points
        t_start, t_end = self.time_span
        t_eval_jax = jnp.linspace(t_start, t_end, self.n_steps)
        t_eval_jax = jax.device_put(t_eval_jax, self.jax_device)

        # Check cache if enabled
        cache_key = None
        if self.use_cache and self._cache_manager is not None:
            solver_config = self._get_cache_config()
            # Use original torch tensors for cache key
            t_eval_torch_cpu = torch.linspace(
                float(t_start), float(t_end), self.n_steps, device="cpu"
            )
            y0_cpu = y0.detach().cpu()

            cache_key = self._cache_manager.build_key(
                self.__class__.__name__,
                ode_system,  # type: ignore[arg-type]  # JaxODESystem has get_str method
                y0_cpu,
                t_eval_torch_cpu,
                solver_config,
            )
            cached_result = self._cache_manager.load(cache_key, self.device)

            if cached_result is not None:
                print(f"    [{self.__class__.__name__}] Loaded result from cache")
                return cached_result

        # Compute integration
        if self.use_cache:
            print(f"    [{self.__class__.__name__}] Cache miss - integrating...")
        else:
            print(f"    [{self.__class__.__name__}] Cache disabled - integrating...")

        # Cast to concrete type for internal implementation
        ode_system_concrete = cast(JaxODESystem[Any], ode_system)
        t_result_jax, y_result_jax = self._integrate_jax(ode_system_concrete, y0_jax, t_eval_jax)
        print(f"    [{self.__class__.__name__}] Integration complete")

        # Convert back to PyTorch
        torch_device = str(y0.device)
        t_result = jax_to_torch(t_result_jax, torch_device)
        y_result = jax_to_torch(y_result_jax, torch_device)

        # Save to cache if enabled
        if self.use_cache and cache_key is not None and self._cache_manager is not None:
            self._cache_manager.save(cache_key, t_result.cpu(), y_result.cpu())

        return t_result, y_result

    def _integrate_jax(
        self, ode_system: JaxODESystem[Any], y0: Array, t_eval: Array
    ) -> tuple[Array, Array]:
        """
        Perform the actual integration using JAX/Diffrax.

        :param ode_system: An instance of JaxODESystem.
        :param y0: Initial conditions as JAX array with shape (batch, n_dims).
        :param t_eval: Time points as JAX array.
        :return: (t_eval, y_values) as JAX arrays.
        """
        # Get the raw ODE function from the system for better JIT tracing
        ode_fn = ode_system.ode

        def ode_wrapper(t: Any, y: Array, args: Any) -> Array:
            return ode_fn(t, y)  # type: ignore[arg-type]

        term = ODETerm(ode_wrapper)
        t0, t1 = float(t_eval[0]), float(t_eval[-1])
        saveat = SaveAt(ts=t_eval)
        stepsize_controller = PIDController(rtol=self.rtol, atol=self.atol)

        # Create event if event_fn is provided
        event = Event(cond_fn=self.event_fn) if self.event_fn is not None else None

        # Define solve function for a single initial condition
        def solve_single(y0_single: Array) -> Array:
            sol = diffeqsolve(  # type: ignore[misc]
                term,
                self.diffrax_solver,
                t0=t0,
                t1=t1,
                dt0=None,
                y0=y0_single,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                max_steps=self.max_steps,
                event=event,
            )
            return sol.ys  # type: ignore[return-value]

        # JIT compile and vmap for batch processing
        solve_batch = jax.jit(jax.vmap(solve_single))

        try:
            # Integrate all trajectories in parallel
            y_batch: Array = solve_batch(y0)
            jax.block_until_ready(y_batch)  # type: ignore[no-untyped-call]
        except Exception as e:
            raise RuntimeError(f"JAX/Diffrax integration failed: {e}") from e

        # Transpose from (batch, n_steps, n_dims) to (n_steps, batch, n_dims)
        y_batch_transposed: Array = jnp.transpose(y_batch, (1, 0, 2))  # type: ignore[arg-type]

        return t_eval, y_batch_transposed
