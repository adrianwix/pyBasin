from abc import ABC, abstractmethod

import numpy as np
import torch
import torchode as to
from scipy.integrate import solve_ivp
from sklearn.utils.parallel import Parallel, delayed
from torchdiffeq import odeint

from pybasin.cache_manager import CacheManager
from pybasin.ode_system import ODESystem
from pybasin.utils import resolve_folder


class Solver(ABC):
    """Abstract base class for ODE solvers with persistent caching.

    The cache is stored both in-memory and on disk.
    The cache key is built using:
      - The solver class name,
      - The solver-specific configuration (rtol, atol, method, etc.),
      - The ODE system's string representation via ode_system.get_str(),
      - The serialized initial conditions (y0),
      - The serialized evaluation time points (t_eval).
    The persistent cache is stored in the folder given by resolve_folder("cache").
    """

    def __init__(
        self,
        time_span: tuple[float, float],
        fs: float | None = None,
        n_steps: int | None = None,
        device: str | None = None,
        use_cache: bool = True,
        **kwargs,
    ):
        """
        Initialize the solver with integration parameters.

        :param time_span: Tuple (t_start, t_end) defining the integration interval.
        :param fs: Sampling frequency (Hz) - number of samples per time unit. DEPRECATED: use n_steps instead.
        :param n_steps: Number of evaluation points. If None, defaults to 500 (recommended for most cases).
        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        :param use_cache: Whether to use caching for integration results (default: True).
        """
        self.time_span = time_span

        # Handle n_steps with backward compatibility
        if n_steps is not None:
            self.n_steps = n_steps
        elif fs is not None:
            # Legacy behavior: compute from fs (not recommended)
            self.n_steps = int((time_span[1] - time_span[0]) * fs) + 1
            print(
                f"Warning: Using fs={fs} results in {self.n_steps} steps. Consider using n_steps parameter directly."
            )
        else:
            # Default: use 500 evaluation points (sufficient for most ODEs)
            self.n_steps = 500

        self.fs = fs  # Keep for backward compatibility
        # TODO: Review if params is necessary
        self.params = kwargs  # Additional solver parameters
        self.use_cache = use_cache

        # Auto-detect device if not specified and normalize cuda to cuda:0
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            # Normalize "cuda" to "cuda:0" for consistency
            dev = torch.device(device)
            if dev.type == "cuda" and dev.index is None:
                self.device = torch.device("cuda:0")
            else:
                self.device = dev

        self._cache_manager = CacheManager(resolve_folder("cache"))

    def _prepare_tensors(self, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare time evaluation points and initial conditions with correct device and dtype."""
        t_start, t_end = self.time_span
        # Use float32 for GPU efficiency (5-10x faster than float64 on consumer GPUs)
        t_eval = torch.linspace(
            t_start, t_end, self.n_steps, dtype=torch.float32, device=self.device
        )

        # Warn if y0 is on wrong device or has wrong dtype
        if y0.device != self.device:
            print(f"  Warning: y0 is on device {y0.device} but solver expects {self.device}")
        if y0.dtype != torch.float32:
            print(f"  Warning: y0 has dtype {y0.dtype} but solver expects torch.float32")

        return t_eval, y0

    def integrate(
        self, ode_system: ODESystem, y0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the ODE system and return the evaluation time points and solution.
        Uses caching to avoid recomputation if the same problem was solved before.

        :param ode_system: An instance of ODESystem.
        :param y0: Initial conditions with shape (batch, n_dims) where batch is the number
                   of initial conditions and n_dims is the number of state variables.
        :return: Tuple (t_eval, y_values) where y_values has shape (n_steps, batch, n_dims).
        """
        # Validate y0 shape
        if y0.ndim != 2:
            raise ValueError(
                f"y0 must be 2D with shape (batch, n_dims), got shape {y0.shape}. "
                f"For single initial condition, use y0.unsqueeze(0) or y0.reshape(1, -1)."
            )

        # Prepare tensors with correct device and dtype
        t_eval, y0 = self._prepare_tensors(y0)

        # Move ODE system to the correct device
        ode_system = ode_system.to(self.device)

        # Check cache if enabled
        cache_key = None

        if self.use_cache:
            solver_config = self._get_cache_config()
            cache_key = self._cache_manager.build_key(
                self.__class__.__name__, ode_system, y0, t_eval, solver_config
            )
            cached_result = self._cache_manager.load(cache_key, self.device)

            if cached_result is not None:
                print(f"    [{self.__class__.__name__}] Loaded result from cache")
                return cached_result

        # Compute integration if not cached or cache disabled
        if self.use_cache:
            print(f"    [{self.__class__.__name__}] Cache miss - integrating on {self.device}...")
        else:
            print(
                f"    [{self.__class__.__name__}] Cache disabled - integrating on {self.device}..."
            )
        t_result, y_result = self._integrate(ode_system, y0, t_eval)
        print(f"    [{self.__class__.__name__}] Integration complete")

        # Save to cache if enabled
        if self.use_cache and cache_key is not None:
            self._cache_manager.save(cache_key, t_result, y_result)

        return t_result, y_result

    def _get_cache_config(self) -> dict:
        """
        Get solver-specific configuration for cache key.
        Subclasses should override this to include relevant parameters.

        :return: Dictionary of configuration parameters that affect integration results.
        """
        return {}

    @abstractmethod
    def _integrate(
        self, ode_system: ODESystem, y0: torch.Tensor, t_eval: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the actual integration using the given solver.
        This method is implemented by subclasses.

        :param ode_system: An instance of ODESystem.
        :param y0: Initial conditions.
        :param t_eval: Time points at which the solution is evaluated.
        :return: (t_eval, y_values)
        """
        pass


class TorchDiffEqSolver(Solver):
    """
    Solver using torchdiffeq's odeint.
    """

    def __init__(
        self,
        time_span,
        fs=None,
        n_steps=None,
        device=None,
        method: str = "dopri5",
        rtol: float = 1e-8,
        atol: float = 1e-6,
        **kwargs,
    ):
        """
        Initialize TorchDiffEqSolver.

        :param time_span: Tuple (t_start, t_end) defining the integration interval.
        :param fs: Sampling frequency (Hz). DEPRECATED: use n_steps instead.
        :param n_steps: Number of evaluation points. If None, defaults to 500.
        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        :param method: Integration method from tordiffeq.odeint.
        :param rtol: Relative tolerance for adaptive stepping.
        :param atol: Absolute tolerance for adaptive stepping.
        """
        super().__init__(time_span, fs=fs, n_steps=n_steps, device=device, **kwargs)
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def _get_cache_config(self) -> dict:
        """Include method, rtol, and atol in cache key."""
        return {"method": self.method, "rtol": self.rtol, "atol": self.atol}

    def _integrate(
        self, ode_system: ODESystem, y0: torch.Tensor, t_eval: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate using torchdiffeq's odeint.

        :param ode_system: An instance of ODESystem.
        :param y0: Initial conditions with shape (batch, n_dims).
        :param t_eval: Time points at which the solution is evaluated (1D tensor).
        :return: (t_eval, y_values) where y_values has shape (n_steps, batch, n_dims).
        """
        try:
            # Type checker incorrectly infers tuple return type, but runtime is torch.Tensor
            y_torch: torch.Tensor = odeint(
                ode_system, y0, t_eval, method=self.method, rtol=self.rtol, atol=self.atol
            )  # type: ignore
        except RuntimeError as e:
            raise e
        return t_eval, y_torch


class TorchOdeSolver(Solver):
    """
    Solver using torchode's parallel ODE solver.
    """

    def __init__(
        self,
        time_span: tuple[float, float],
        fs: float | None = None,
        n_steps: int | None = None,
        device: str | None = None,
        method: str = "dopri5",
        rtol: float = 1e-8,
        atol: float = 1e-6,
        use_jit: bool = False,
        **kwargs,
    ):
        """
        Initialize TorchOdeSolver.

        :param time_span: Tuple (t_start, t_end) defining the integration interval.
        :param fs: Sampling frequency (Hz). DEPRECATED: use n_steps instead.
        :param n_steps: Number of evaluation points. If None, defaults to 500.
        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        :param method: Integration method ('dopri5', 'tsit5', 'euler', 'heun').
        :param rtol: Relative tolerance for adaptive stepping.
        :param atol: Absolute tolerance for adaptive stepping.
        :param use_jit: Whether to use JIT compilation (can improve performance).
        """
        super().__init__(time_span, fs=fs, n_steps=n_steps, device=device, **kwargs)
        self.method = method.lower()
        self.rtol = rtol
        self.atol = atol
        self.use_jit = use_jit

    def _get_cache_config(self) -> dict:
        """Include method, rtol, atol, and use_jit in cache key."""
        return {
            "method": self.method,
            "rtol": self.rtol,
            "atol": self.atol,
            "use_jit": self.use_jit,
        }

    def _integrate(
        self, ode_system: ODESystem, y0: torch.Tensor, t_eval: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate using torchode.

        :param ode_system: An instance of ODESystem.
        :param y0: Initial conditions with shape (batch, n_dims).
        :param t_eval: Time points at which the solution is evaluated (1D tensor).
        :return: (t_eval, y_values) where y_values has shape (n_steps, batch, n_dims).
        """
        # y0 is guaranteed to be 2D (batch, n_dims) by Solver.integrate validation
        y0_batched = y0
        batch_size = y0.shape[0]

        # For torchode, we need t_start and t_end as (batch,) tensors
        # Repeat for each sample in the batch
        t_start = torch.full(
            (batch_size,), t_eval[0].item(), device=t_eval.device, dtype=t_eval.dtype
        )
        t_end = torch.full(
            (batch_size,), t_eval[-1].item(), device=t_eval.device, dtype=t_eval.dtype
        )

        # t_eval needs to be (batch, n_steps) - repeat for each sample
        t_eval_batched = t_eval.unsqueeze(0).expand(batch_size, -1) if t_eval.ndim == 1 else t_eval

        # Create ODE function wrapper for torchode
        # torchode calls f(t, y) where t is scalar and y is (batch, n_dims)
        def torchode_func(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # y shape: (batch, n_dims)
            # Call ODE system which handles batched input correctly
            return ode_system(t, y)

        # Create torchode components
        term = to.ODETerm(torchode_func)  # pyright: ignore[reportArgumentType]

        # Select step method
        if self.method == "dopri5":
            step_method = to.Dopri5(term=term)
        elif self.method == "tsit5":
            step_method = to.Tsit5(term=term)
        elif self.method == "euler":
            step_method = to.Euler(term=term)
        elif self.method == "heun":
            step_method = to.Heun(term=term)
        else:
            raise ValueError(
                f"Unknown method: {self.method}. Available: dopri5, tsit5, euler, midpoint, heun"
            )

        step_size_controller = to.IntegralController(atol=self.atol, rtol=self.rtol, term=term)
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)  # pyright: ignore[reportArgumentType]

        # Create initial value problem with matching batch sizes
        problem = to.InitialValueProblem(
            y0=y0_batched,  # pyright: ignore[reportArgumentType]
            t_start=t_start,  # pyright: ignore[reportArgumentType]
            t_end=t_end,  # pyright: ignore[reportArgumentType]
            t_eval=t_eval_batched,  # pyright: ignore[reportArgumentType]
        )

        # Solve
        try:
            solution = solver.solve(problem)
        except RuntimeError as e:
            raise RuntimeError(f"torchode integration failed: {e}") from e

        # Extract solution and transpose to match expected format
        # torchode returns (batch, n_steps, n_dims)
        # We need (n_steps, batch, n_dims) to match TorchDiffEqSolver
        y_result = solution.ys.transpose(0, 1)

        return t_eval, y_result


class SklearnParallelSolver(Solver):
    """
    ODE solver using sklearn's parallel processing with scipy's solve_ivp.

    Uses multiprocessing (loky backend) to solve multiple initial conditions in parallel.
    Each worker solves one trajectory at a time using scipy's solve_ivp.
    """

    def __init__(
        self,
        time_span: tuple[float, float],
        fs: float,
        device: str | None = None,
        n_jobs: int = -1,
        batch_size: int | None = None,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_step: float | None = None,
        **kwargs,
    ):
        """
        Initialize SklearnParallelSolver.

        :param time_span: Integration interval (t_start, t_end).
        :param fs: Sampling frequency (Hz).
        :param device: Device to use (only 'cpu' supported).
        :param n_jobs: Number of parallel jobs (-1 for all CPUs).
        :param batch_size: Unused, kept for API compatibility.
        :param method: Integration method ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA', etc).
        :param rtol: Relative tolerance for the solver.
        :param atol: Absolute tolerance for the solver.
        :param max_step: Maximum step size for the solver.
        """
        if device and "cuda" in device:
            print("  Warning: SklearnParallelSolver does not support CUDA - falling back to CPU")
            device = "cpu"

        super().__init__(time_span, fs, device="cpu", **kwargs)

        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step or (time_span[1] - time_span[0]) / 100

    def _get_cache_config(self) -> dict:
        """Include method, rtol, atol, and max_step in cache key."""
        return {
            "method": self.method,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_step": self.max_step,
        }

    def _integrate(
        self, ode_system: ODESystem, y0: torch.Tensor, t_eval: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate using sklearn parallel processing with scipy's solve_ivp."""
        t_eval_np = t_eval.cpu().numpy()
        y0_np = y0.cpu().numpy()

        if y0_np.ndim == 1:
            y0_np = y0_np.reshape(1, -1)
            single_trajectory = True
        else:
            single_trajectory = False

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
            solution = solve_ivp(
                fun=lambda t, y: ode_func(t, y),
                t_span=(t_eval_np[0], t_eval_np[-1]),
                y0=y0_single,
                method=self.method,
                t_eval=t_eval_np,
                rtol=self.rtol,
                atol=self.atol,
                max_step=self.max_step,
            )
            return solution.y.T

        if batch_size == 1 or self.n_jobs == 1:
            results = [solve_single_trajectory(y0_np[0])]
        else:
            results = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=0)(
                delayed(solve_single_trajectory)(y0_np[i])  # type: ignore
                for i in range(batch_size)
            )

        y_result_np = (
            results[0] if single_trajectory else np.stack(results, axis=1)  # type: ignore[arg-type]
        )
        y_result = torch.tensor(y_result_np, dtype=torch.float32, device=self.device)

        return t_eval, y_result
