# pyright: reportUntypedBaseClass=false
from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchode as to  # type: ignore[import-untyped]
from diffrax import (
    ODETerm,
    PIDController,
    SaveAt,
    Tsit5,
    diffeqsolve,  # type: ignore[import-untyped]
)
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
            # For CUDA devices, normalize to cuda:0 if no specific index given
            if dev.type == "cuda":
                # If device string was just "cuda" without index, index will be None
                # torch.device("cuda").index returns None, not 0
                idx = dev.index if dev.index is not None else 0  # pyright: ignore[reportUnnecessaryComparison]
                self.device = torch.device(f"cuda:{idx}")
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


class JaxSolver(Solver):
    """
    ODE solver using JAX and Diffrax for high-performance GPU integration.

    This solver leverages JAX's JIT compilation and automatic differentiation
    with Diffrax's adaptive ODE solvers. It uses vmap for efficient batch processing
    and supports both CPU and GPU devices.
    """

    def __init__(
        self,
        time_span: tuple[float, float],
        fs: float | None = None,
        n_steps: int | None = None,
        device: str | None = None,
        solver: Any | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_steps: int = 16**5,
        **kwargs: Any,
    ):
        """
        Initialize JaxSolver.

        :param time_span: Tuple (t_start, t_end) defining the integration interval.
        :param fs: Sampling frequency (Hz). DEPRECATED: use n_steps instead.
        :param n_steps: Number of evaluation points. If None, defaults to 500.
        :param device: Device to use ('cuda', 'gpu', 'cpu', or None for auto-detect).
        :param solver: Diffrax solver instance (e.g., Tsit5(), Dopri5()). If None, defaults to Tsit5().
        :param rtol: Relative tolerance for adaptive stepping.
        :param atol: Absolute tolerance for adaptive stepping.
        :param max_steps: Maximum number of steps for the integrator.
        """
        # Map device strings to JAX device types
        jax_device: Any
        if device is None:
            # Auto-detect: prefer GPU if available
            jax_device = jax.devices()[0]  # type: ignore[misc]
        elif device in ["cuda", "gpu", "cuda:0"]:
            # Try to get GPU device
            gpu_devices: list[Any] = [d for d in jax.devices() if d.device_kind == "gpu"]  # type: ignore[misc, union-attr]
            if gpu_devices:
                jax_device = gpu_devices[0]
            else:
                print("  Warning: GPU requested but not available - falling back to CPU")
                jax_device = jax.devices("cpu")[0]  # type: ignore[misc]
        elif device == "cpu":
            jax_device = jax.devices("cpu")[0]  # type: ignore[misc]
        else:
            # Try to parse as specific device
            jax_device = jax.devices()[0]  # type: ignore[misc]

        # Store JAX device
        self.jax_device: Any = jax_device  # JAX Device type not fully typed

        # Call parent with normalized device string for PyTorch compatibility
        device_str = "cuda:0" if jax_device.device_kind == "gpu" else "cpu"  # type: ignore[union-attr]
        super().__init__(time_span, fs=fs, n_steps=n_steps, device=device_str, **kwargs)

        # Use provided solver or default to Tsit5
        self.diffrax_solver = solver if solver is not None else Tsit5()
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

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
        Integrate using JAX/Diffrax.

        :param ode_system: An instance of ODESystem.
        :param y0: Initial conditions with shape (batch, n_dims).
        :param t_eval: Time points at which the solution is evaluated (1D tensor).
        :return: (t_eval, y_values) where y_values has shape (n_steps, batch, n_dims).
        """
        # Convert torch tensors to JAX arrays on the appropriate device
        y0_jax: Any = jax.device_put(  # pyright: ignore[reportUnknownMemberType]
            jnp.array(y0.cpu().numpy(), dtype=jnp.float32),  # pyright: ignore[reportUnknownMemberType]
            self.jax_device,  # type: ignore[misc]
        )
        t_eval_jax: Any = jax.device_put(  # type: ignore[reportUnknownMemberType]
            jnp.array(t_eval.cpu().numpy(), dtype=jnp.float32),  # type: ignore[reportUnknownMemberType]
            self.jax_device,
        )

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

        # Convert back to PyTorch tensors
        y_result = torch.tensor(np.array(y_batch_jax), dtype=torch.float32, device=self.device)

        return t_eval, y_result
