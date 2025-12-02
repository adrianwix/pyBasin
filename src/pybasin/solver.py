# pyright: reportUntypedBaseClass=false
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torchode as to  # type: ignore[import-untyped]
from scipy.integrate import solve_ivp
from sklearn.utils.parallel import Parallel, delayed  # type: ignore[import-untyped]
from torchdiffeq import odeint  # type: ignore[import-untyped]

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

    display_name: str = "Solver"

    def __init__(
        self,
        time_span: tuple[float, float],
        fs: float | None = None,
        n_steps: int | None = None,
        device: str | None = None,
        use_cache: bool = True,
        **kwargs: Any,
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
        self.fs = fs  # Keep for backward compatibility
        self.params = kwargs  # Additional solver parameters
        self.use_cache = use_cache

        self._set_n_steps(n_steps, fs)
        self._set_device(device)

        # Only create cache manager if caching is enabled
        # This avoids resolve_folder issues when called from threads
        self._cache_manager: CacheManager | None = None
        if use_cache:
            self._cache_manager = CacheManager(resolve_folder("cache"))

    def _set_n_steps(self, n_steps: int | None, fs: float | None) -> None:
        """
        Set the number of evaluation steps with backward compatibility for fs parameter.

        :param n_steps: Number of evaluation points. If provided, used directly.
        :param fs: Sampling frequency (Hz). DEPRECATED: used only if n_steps is None.
        """
        if n_steps is not None:
            self.n_steps = n_steps
        elif fs is not None:
            # Legacy behavior: compute from fs (not recommended)
            self.n_steps = int((self.time_span[1] - self.time_span[0]) * fs) + 1
            print(
                f"Warning: Using fs={fs} results in {self.n_steps} steps. Consider using n_steps parameter directly."
            )
        else:
            # Default: use 500 evaluation points (sufficient for most ODEs)
            self.n_steps = 500

    def _set_device(self, device: str | None) -> None:
        """
        Set the device for tensor operations with auto-detection and normalization.

        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        # Store original device string for with_device()
        self._device_str = device

        # Auto-detect device if not specified and normalize cuda to cuda:0
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            # Normalize "cuda" to "cuda:0" for consistency
            dev = torch.device(device)
            # For CUDA devices, normalize to cuda:0 if no specific index given
            if dev.type == "cuda":
                # If device string was just "cuda" without index, index will be None
                # torch.device("cuda").index returns None, not 0
                idx = dev.index if dev.index is not None else 0  # pyright: ignore[reportUnnecessaryComparison]
                self.device = torch.device(f"cuda:{idx}")
            else:
                self.device = dev

    @abstractmethod
    def with_device(self, device: str) -> "Solver":
        """
        Create a copy of this solver configured for a different device.

        :param device: Target device ('cpu', 'cuda').
        :return: New solver instance with the same configuration but different device.
        """
        pass

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
        self, ode_system: ODESystem[Any], y0: torch.Tensor
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

        if self.use_cache and self._cache_manager is not None:
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
        if self.use_cache and cache_key is not None and self._cache_manager is not None:
            self._cache_manager.save(cache_key, t_result, y_result)

        return t_result, y_result

    def _get_cache_config(self) -> dict[str, Any]:
        """
        Get solver-specific configuration for cache key.
        Subclasses should override this to include relevant parameters.

        :return: Dictionary of configuration parameters that affect integration results.
        """
        return {}

    @abstractmethod
    def _integrate(
        self, ode_system: ODESystem[Any], y0: torch.Tensor, t_eval: torch.Tensor
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

    display_name: str = "TorchDiffEq Solver"

    def __init__(
        self,
        time_span: tuple[float, float],
        fs: float | None = None,
        n_steps: int | None = None,
        device: str | None = None,
        method: str = "dopri5",
        rtol: float = 1e-8,
        atol: float = 1e-6,
        **kwargs: Any,
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

    def _get_cache_config(self) -> dict[str, Any]:
        """Include method, rtol, and atol in cache key."""
        return {"method": self.method, "rtol": self.rtol, "atol": self.atol}

    def with_device(self, device: str) -> "TorchDiffEqSolver":
        """Create a copy of this solver configured for a different device."""
        return TorchDiffEqSolver(
            time_span=self.time_span,
            fs=self.fs,
            n_steps=self.n_steps,
            device=device,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            use_cache=self.use_cache,
            **self.params,
        )

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
        try:
            # odeint returns Tensor, but type stubs are incomplete
            y_torch: torch.Tensor = odeint(  # type: ignore[assignment]
                ode_system, y0, t_eval, method=self.method, rtol=self.rtol, atol=self.atol
            )
        except RuntimeError as e:
            raise e
        return t_eval, y_torch


class TorchOdeSolver(Solver):
    """
    Solver using torchode's parallel ODE solver.
    """

    display_name: str = "TorchODE Solver"

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
        **kwargs: Any,
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

    def _get_cache_config(self) -> dict[str, Any]:
        """Include method, rtol , atol, and use_jit in cache key."""
        return {
            "method": self.method,
            "rtol": self.rtol,
            "atol": self.atol,
            "use_jit": self.use_jit,
        }

    def with_device(self, device: str) -> "TorchOdeSolver":
        """Create a copy of this solver configured for a different device."""
        return TorchOdeSolver(
            time_span=self.time_span,
            fs=self.fs,
            n_steps=self.n_steps,
            device=device,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            use_jit=self.use_jit,
            use_cache=self.use_cache,
            **self.params,
        )

    def _integrate(
        self, ode_system: ODESystem[Any], y0: torch.Tensor, t_eval: torch.Tensor
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

    display_name: str = "Sklearn Parallel Solver"

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
        **kwargs: Any,
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

    def _get_cache_config(self) -> dict[str, Any]:
        """Include method, rtol, atol, and max_step in cache key."""
        return {
            "method": self.method,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_step": self.max_step,
        }

    def with_device(self, device: str) -> "SklearnParallelSolver":
        """Create a copy of this solver configured for a different device.

        Note: SklearnParallelSolver only supports CPU, so this always returns a CPU solver.
        """
        if device and "cuda" in device:
            print("  Warning: SklearnParallelSolver does not support CUDA - using CPU")
        return SklearnParallelSolver(
            time_span=self.time_span,
            fs=self.fs,  # type: ignore[arg-type]
            device="cpu",
            n_jobs=self.n_jobs,
            batch_size=self.batch_size,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step,
            use_cache=self.use_cache,
            **self.params,
        )

    def _integrate(
        self, ode_system: ODESystem[Any], y0: torch.Tensor, t_eval: torch.Tensor
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

        # Stack results into array
        if single_trajectory:
            y_result_np: np.ndarray = results[0]  # type: ignore[index]
        else:
            # Filter out None values and stack
            valid_results = [r for r in results if r is not None]  # type: ignore[misc]
            y_result_np = np.stack(valid_results, axis=1)  # type: ignore[arg-type]

        y_result = torch.tensor(y_result_np, dtype=torch.float32, device=self.device)

        return t_eval, y_result
