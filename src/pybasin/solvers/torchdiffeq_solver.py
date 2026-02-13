import logging
from typing import Any

import torch
from torchdiffeq import odeint  # type: ignore[import-untyped]

from pybasin.cache_manager import CacheManager
from pybasin.ode_system import ODESystem
from pybasin.solvers.base import Solver

logger = logging.getLogger(__name__)


class TorchDiffEqSolver(Solver):
    """
    Differentiable ODE solver with full GPU support and O(1)-memory backpropagation.

    Uses the adjoint method for memory-efficient gradient computation through ODE solutions.
    Supports adaptive-step (dopri5, dopri8, bosh3) and fixed-step (euler, rk4) methods.

    See also: [torchdiffeq GitHub](https://github.com/rtqichen/torchdiffeq)

    Citation:

    ```bibtex
    @misc{torchdiffeq,
        author={Chen, Ricky T. Q.},
        title={torchdiffeq},
        year={2018},
        url={https://github.com/rtqichen/torchdiffeq},
    }
    ```
    """

    def __init__(
        self,
        time_span: tuple[float, float],
        n_steps: int | None = None,
        device: str | None = None,
        method: str = "dopri5",
        rtol: float = 1e-8,
        atol: float = 1e-6,
        use_cache: bool = True,
    ):
        """
        Initialize TorchDiffEqSolver.

        :param time_span: Tuple (t_start, t_end) defining the integration interval.
        :param n_steps: Number of evaluation points. If None, defaults to 1000.
        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        :param method: Integration method from tordiffeq.odeint.
        :param rtol: Relative tolerance for adaptive stepping.
        :param atol: Absolute tolerance for adaptive stepping.
        :param use_cache: Whether to use caching for integration results.
        """
        super().__init__(
            time_span, n_steps=n_steps, device=device, rtol=rtol, atol=atol, use_cache=use_cache
        )
        self.method = method

    def _get_cache_config(self) -> dict[str, Any]:
        """Include method in cache key (rtol/atol handled by base class)."""
        config = super()._get_cache_config()
        config["method"] = self.method
        return config

    def clone(
        self,
        *,
        device: str | None = None,
        n_steps_factor: int = 1,
        use_cache: bool | None = None,
    ) -> "TorchDiffEqSolver":
        """Create a copy of this solver, optionally overriding device, resolution, or caching."""
        new_solver = TorchDiffEqSolver(
            time_span=self.time_span,
            n_steps=self.n_steps * n_steps_factor,
            device=device or self._device_str,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            use_cache=use_cache if use_cache is not None else self.use_cache,
        )
        if self._cache_dir is not None:
            new_solver._cache_dir = self._cache_dir
            new_solver._cache_manager = CacheManager(self._cache_dir)
        return new_solver

    def _integrate(
        self, ode_system: ODESystem[Any], y0: torch.Tensor, t_eval: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate using torchdiffeq's odeint.

        :param ode_system: An instance of ODESystem.
        :param y0: Initial conditions with shape (batch, n_dims).
        :param t_eval: Time points at which the solution is evaluated (1D tensor).
        :return: (t_eval, y_values) where y_values has shape (n_steps, batch, n_dims).
        """
        # odeint returns Tensor, but type stubs are incomplete
        with torch.no_grad():
            y_torch: torch.Tensor = odeint(  # type: ignore[assignment]
                ode_system, y0, t_eval, method=self.method, rtol=self.rtol, atol=self.atol
            )
        return t_eval, y_torch
