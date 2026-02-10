# pyright: reportUntypedBaseClass=false
"""JAX solver for PyTorch ODE systems.

This module provides a JAX/Diffrax solver that can integrate PyTorch-based ODE systems
by using JAX callbacks. This approach is slower than native JAX ODE systems but allows
using existing PyTorch ODE definitions.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from diffrax import (
    Dopri5,
    ODETerm,
    PIDController,
    SaveAt,
    diffeqsolve,  # type: ignore[import-untyped]
)

from pybasin.cache_manager import CacheManager
from pybasin.jax_utils import get_jax_device, jax_to_torch, torch_to_jax
from pybasin.ode_system import ODESystem
from pybasin.solver import Solver


class JaxForPytorchSolver(Solver):
    """
    ODE solver using JAX and Diffrax for PyTorch-based ODE systems.

    This solver wraps PyTorch ODE systems using JAX callbacks, which adds overhead
    compared to native JAX ODE systems. For maximum performance, use JaxSolver
    with JaxODESystem instead.

    This solver leverages JAX's JIT compilation and automatic differentiation
    with Diffrax's adaptive ODE solvers. It uses vmap for efficient batch processing
    and supports both CPU and GPU devices.

    Note
    ----
    For better performance, consider using JaxSolver with JaxODESystem
    which avoids the PyTorch callback overhead.
    """

    def __init__(
        self,
        time_span: tuple[float, float],
        n_steps: int | None = None,
        device: str | None = None,
        solver: Any | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_steps: int = 16**5,
        **kwargs: Any,
    ):
        """
        Initialize JaxForPytorchSolver.

        :param time_span: Tuple (t_start, t_end) defining the integration interval.
        :param n_steps: Number of evaluation points. If None, defaults to 1000.
        :param device: Device to use ('cuda', 'gpu', 'cpu', or None for auto-detect).
        :param solver: Diffrax solver instance (e.g., Dopri5(), Tsit5()). If None, defaults to Dopri5().
        :param rtol: Relative tolerance for adaptive stepping.
        :param atol: Absolute tolerance for adaptive stepping.
        :param max_steps: Maximum number of steps for the integrator.
        """
        # Use centralized device resolution from jax_utils
        self.jax_device: Any = get_jax_device(device)

        # Call parent with normalized device string for PyTorch compatibility
        device_str = "cuda:0" if self.jax_device.platform == "gpu" else "cpu"
        super().__init__(time_span, n_steps=n_steps, device=device_str, **kwargs)

        # Use provided solver or default to Dopri5 (similar to MATLAB's ode45)
        self.diffrax_solver = solver if solver is not None else Dopri5()
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

    def with_device(self, device: str) -> "JaxForPytorchSolver":
        """Create a copy of this solver configured for a different device."""
        new_solver = JaxForPytorchSolver(
            time_span=self.time_span,
            n_steps=self.n_steps,
            device=device,
            solver=self.diffrax_solver,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
            use_cache=self.use_cache,
            **self.params,
        )
        # Reuse the same cache directory to ensure consistency
        if self._cache_dir is not None:
            new_solver._cache_dir = self._cache_dir
            new_solver._cache_manager = CacheManager(self._cache_dir)
        return new_solver

    def _get_cache_config(self) -> dict[str, Any]:
        """Include solver type, rtol, atol, and max_steps in cache key."""
        return {
            "solver": self.diffrax_solver.__class__.__name__,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_steps": self.max_steps,
        }

    def _integrate(
        self, ode_system: ODESystem[Any], y0: torch.Tensor, t_eval: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate using JAX/Diffrax with PyTorch ODE callback.

        :param ode_system: An instance of ODESystem (PyTorch-based).
        :param y0: Initial conditions with shape (batch, n_dims).
        :param t_eval: Time points at which the solution is evaluated (1D tensor).
        :return: (t_eval, y_values) where y_values has shape (n_steps, batch, n_dims).
        """
        # Convert torch tensors to JAX arrays using utilities (zero-copy on GPU via DLPack)
        y0_jax: Any = torch_to_jax(y0, self.jax_device)
        t_eval_jax: Any = torch_to_jax(t_eval, self.jax_device)

        # Define ODE function for Diffrax using pure JAX operations
        # Diffrax expects f(t, y, args) where t is scalar and y matches y0 shape
        def diffrax_ode(t: Any, y: Any, args: Any) -> jax.Array:
            # Use JAX callback to call PyTorch ODE system
            # This allows us to use the existing PyTorch ODE definitions
            def torch_ode_callback(t_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
                # Convert to torch tensors (happens outside JAX tracing)
                t_torch = torch.tensor(t_val, dtype=torch.float32, device=self.device)
                y_torch = torch.tensor(y_val, dtype=torch.float32, device=self.device)

                # Ensure y_torch is 2D: (1, n_dims) for single trajectory
                if y_torch.ndim == 1:
                    y_torch = y_torch.unsqueeze(0)

                # Call ODE system
                dy_torch = ode_system(t_torch, y_torch)

                # Convert back to numpy and squeeze batch dimension
                if dy_torch.ndim == 2 and dy_torch.shape[0] == 1:
                    dy_torch = dy_torch.squeeze(0)

                return dy_torch.cpu().numpy()

            # Use pure_callback to call PyTorch code from JAX
            result_shape = jax.ShapeDtypeStruct(y.shape, jnp.float32)
            dy_jax = jax.pure_callback(
                torch_ode_callback, result_shape, t, y, vmap_method="sequential"
            )
            return dy_jax

        # Setup Diffrax components
        term = ODETerm(diffrax_ode)
        t0, t1 = float(t_eval_jax[0]), float(t_eval_jax[-1])
        saveat = SaveAt(ts=t_eval_jax)
        stepsize_controller = PIDController(rtol=self.rtol, atol=self.atol)

        # Define solve function for a single initial condition
        def solve_single(y0_single: Any) -> jax.Array:
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
            )
            return sol.ys  # type: ignore[return-value]

        # Use vmap for batch processing
        solve_batch = jax.vmap(solve_single)

        try:
            # Integrate all trajectories in parallel
            y_batch_jax = solve_batch(y0_jax)
            jax.block_until_ready(y_batch_jax)  # type: ignore[no-untyped-call]
        except Exception as e:
            raise RuntimeError(f"JAX/Diffrax integration failed: {e}") from e

        # Transpose from (batch, n_steps, n_dims) to (n_steps, batch, n_dims)
        y_batch_jax = jnp.transpose(y_batch_jax, (1, 0, 2))

        # Convert back to PyTorch tensors using utilities (zero-copy on GPU via DLPack)
        y_result = jax_to_torch(y_batch_jax, self.device)

        return t_eval, y_result
