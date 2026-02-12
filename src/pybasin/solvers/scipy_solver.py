# pyright: reportUntypedBaseClass=false
import logging
from typing import Any

import numpy as np
import torch
from scipy.integrate import solve_ivp
from sklearn.utils.parallel import Parallel, delayed  # type: ignore[import-untyped]

from pybasin.cache_manager import CacheManager
from pybasin.ode_system import ODESystem
from pybasin.solvers.base import Solver

logger = logging.getLogger(__name__)


class ScipyParallelSolver(Solver):
    """
    ODE solver using sklearn's parallel processing with scipy's solve_ivp.

    Uses multiprocessing (loky backend) to solve multiple initial conditions in parallel.
    Each worker solves one trajectory at a time using scipy's solve_ivp.

    See also: [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
    """

    def __init__(
        self,
        time_span: tuple[float, float],
        n_steps: int | None = None,
        device: str | None = None,
        n_jobs: int = -1,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_step: float | None = None,
        use_cache: bool = True,
    ):
        """
        Initialize ScipyParallelSolver.

        :param time_span: Integration interval (t_start, t_end).
        :param n_steps: Number of evaluation points. If None, defaults to 1000.
        :param device: Device to use (only 'cpu' supported).
        :param n_jobs: Number of parallel jobs (-1 for all CPUs).
        :param method: Integration method ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA', etc).
        :param rtol: Relative tolerance for the solver.
        :param atol: Absolute tolerance for the solver.
        :param max_step: Maximum step size for the solver.
        :param use_cache: Whether to use caching for integration results.
        """
        if device and "cuda" in device:
            logger.warning(
                "  Warning: ScipyParallelSolver does not support CUDA - falling back to CPU"
            )
            device = "cpu"

        super().__init__(
            time_span, n_steps=n_steps, device="cpu", rtol=rtol, atol=atol, use_cache=use_cache
        )

        self.n_jobs = n_jobs
        self.method = method
        self.max_step = max_step or (time_span[1] - time_span[0]) / 100

    def _get_cache_config(self) -> dict[str, Any]:
        """Include method and max_step in cache key (rtol/atol handled by base class)."""
        config = super()._get_cache_config()
        config["method"] = self.method
        config["max_step"] = self.max_step
        return config

    def with_device(self, device: str) -> "ScipyParallelSolver":
        """Create a copy of this solver configured for a different device.

        Note: ScipyParallelSolver only supports CPU, so this always returns a CPU solver.
        """
        if device and "cuda" in device:
            logger.warning("  Warning: ScipyParallelSolver does not support CUDA - using CPU")
        new_solver = ScipyParallelSolver(
            time_span=self.time_span,
            n_steps=self.n_steps,
            device="cpu",
            n_jobs=self.n_jobs,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step,
            use_cache=self.use_cache,
        )
        # Reuse the same cache directory to ensure consistency
        if self._cache_dir is not None:
            new_solver._cache_dir = self._cache_dir
            new_solver._cache_manager = CacheManager(self._cache_dir)
        return new_solver

    def _integrate(
        self, ode_system: ODESystem[Any], y0: torch.Tensor, t_eval: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate using sklearn parallel processing with scipy's solve_ivp."""
        t_eval_np = t_eval.cpu().numpy()
        y0_np = y0.cpu().numpy()
        batch_size = y0_np.shape[0]

        def ode_func(t: float, y: np.ndarray) -> np.ndarray:
            # Convert to torch, call ODE system, convert back
            t_torch = torch.tensor(t, dtype=torch.float32, device=self.device)
            y_torch = torch.tensor(y, dtype=torch.float32, device=self.device)
            # Ensure y_torch is 2D: (1, n_dims) for ODE system
            if y_torch.ndim == 1:
                y_torch = y_torch.unsqueeze(0)
            dy_torch = ode_system(t_torch, y_torch)
            # Return as 1D array
            if dy_torch.ndim == 2:
                dy_torch = dy_torch.squeeze(0)
            return dy_torch.cpu().numpy()

        # Define solver for single trajectory using scipy's solve_ivp
        def solve_single_trajectory(y0_single: np.ndarray) -> np.ndarray:
            """Solve ODE for a single initial condition using scipy's solve_ivp."""
            # scipy.integrate.solve_ivp has incomplete type stubs
            solution = solve_ivp(  # type: ignore[no-untyped-call]
                fun=lambda t, y: ode_func(float(t), np.asarray(y)),  # type: ignore[arg-type]
                t_span=(t_eval_np[0], t_eval_np[-1]),
                y0=y0_single,
                method=self.method,  # type: ignore[arg-type]
                t_eval=t_eval_np,
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
            )
            return solution.y.T  # type: ignore[no-any-return]

        if batch_size == 1 or self.n_jobs == 1:
            results = [solve_single_trajectory(y0_np[0])]
        else:
            # sklearn.utils.parallel has incomplete type stubs
            results = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=0)(  # type: ignore[misc]
                delayed(solve_single_trajectory)(y0_np[i])  # type: ignore[misc]
                for i in range(batch_size)
            )

        # Filter out None values and stack
        valid_results = [r for r in results if r is not None]  # type: ignore[misc]
        y_result_np: np.ndarray = np.stack(valid_results, axis=1)  # type: ignore[arg-type]

        y_result = torch.tensor(y_result_np, dtype=torch.float32, device=self.device)

        return t_eval, y_result
