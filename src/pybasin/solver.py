from abc import ABC, abstractmethod

import torch
import torchode as to
from torchdiffeq import odeint

from pybasin.cache_manager import CacheManager
from pybasin.ode_system import ODESystem
from pybasin.utils import resolve_folder


class Solver(ABC):
    """Abstract base class for ODE solvers with persistent caching.

    The cache is stored both in-memory and on disk.
    The cache key is built using:
      - The solver class name,
      - The ODE system's string representation via ode_system.get_str(),
      - The serialized initial conditions (y0),
      - The serialized evaluation time points (t_eval).
    The persistent cache is stored in the folder given by resolve_folder("cache").
    """

    def __init__(
        self, time_span: tuple[float, float], fs: float, device: str | None = None, **kwargs
    ):
        """
        Initialize the solver with integration parameters.

        :param time_span: Tuple (t_start, t_end) defining the integration interval.
        :param fs: Sampling frequency (Hz) – number of samples per time unit.
        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        self.time_span = time_span
        self.fs = fs
        self.n_steps = int((time_span[1] - time_span[0]) * fs) + 1
        self.params = kwargs  # Additional solver parameters

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
            print(f"Warning: y0 is on device {y0.device} but solver expects {self.device}")
        if y0.dtype != torch.float32:
            print(f"Warning: y0 has dtype {y0.dtype} but solver expects torch.float32")

        return t_eval, y0

    def integrate(
        self, ode_system: ODESystem, y0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the ODE system and return the evaluation time points and solution.
        Uses caching to avoid recomputation if the same problem was solved before.

        :param ode_system: An instance of ODESystem.
        :param y0: Initial conditions.
        :return: Tuple (t_eval, y_values) where y_values is the solution.
        """
        # Prepare tensors with correct device and dtype
        t_eval, y0 = self._prepare_tensors(y0)

        # Move ODE system to the correct device
        ode_system = ode_system.to(self.device)

        # Check cache
        cache_key = self._cache_manager.build_key(self.__class__.__name__, ode_system, y0, t_eval)
        cached_result = self._cache_manager.load(cache_key, self.device)

        if cached_result is not None:
            print(f"[{self.__class__.__name__}] Loaded result from cache.")
            return cached_result

        # Compute integration if not cached
        print(f"[{self.__class__.__name__}] Cache miss. Integrating on {self.device}...")
        t_result, y_result = self._integrate(ode_system, y0, t_eval)
        print(f"[{self.__class__.__name__}] Integration completed.")

        # Save to cache
        self._cache_manager.save(cache_key, t_result, y_result)

        return t_result, y_result

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
    This class only needs to implement the _integrate method.
    """

    def __init__(self, time_span, fs, **kwargs):
        super().__init__(time_span, fs, **kwargs)

    def _integrate(
        self, ode_system: ODESystem, y0: torch.Tensor, t_eval: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            # odeint returns a torch.Tensor of shape (len(t_eval), len(y0))
            # Type checker incorrectly infers tuple return type, but runtime is torch.Tensor
            y_torch: torch.Tensor = odeint(ode_system, y0, t_eval, rtol=1e-8, atol=1e-6)  # type: ignore
        except RuntimeError as e:
            raise e
        return t_eval, y_torch


class TorchOdeSolver(Solver):
    """
    Solver using torchode's parallel ODE solver.

    torchode provides JIT-compilable ODE solvers that are parallelized across batches.
    This can provide performance benefits, especially with GPU acceleration.

    Available methods:
    - 'dopri5': Dormand-Prince 5(4) (default)
    - 'tsit5': Tsitouras 5(4)
    - 'euler': Explicit Euler
    - 'midpoint': Explicit midpoint
    - 'heun': Heun's method
    """

    def __init__(
        self,
        time_span: tuple[float, float],
        fs: float,
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
        :param fs: Sampling frequency (Hz).
        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        :param method: Integration method ('dopri5', 'tsit5', 'euler', 'heun').
        :param rtol: Relative tolerance for adaptive stepping.
        :param atol: Absolute tolerance for adaptive stepping.
        :param use_jit: Whether to use JIT compilation (can improve performance).
        """
        super().__init__(time_span, fs, device, **kwargs)
        self.method = method.lower()
        self.rtol = rtol
        self.atol = atol
        self.use_jit = use_jit

    def _integrate(
        self, ode_system: ODESystem, y0: torch.Tensor, t_eval: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate using torchode.

        :param ode_system: An instance of ODESystem.
        :param y0: Initial conditions (1D tensor with shape [n_dims]).
        :param t_eval: Time points at which the solution is evaluated (1D tensor).
        :return: (t_eval, y_values) where y_values has shape (n_steps, n_dims).
        """
        # Determine batch size from y0
        if y0.ndim == 1:
            # Single initial condition: (n_dims,)
            y0_batched = y0.unsqueeze(0)  # -> (1, n_dims)
            batch_size = 1
        elif y0.ndim == 2:
            # Already batched: (batch, n_dims)
            y0_batched = y0
            batch_size = y0.shape[0]
        else:
            raise ValueError(f"y0 must be 1D or 2D, got shape {y0.shape}")

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
        term = to.ODETerm(torchode_func)

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
        if batch_size == 1:  # noqa: SIM108
            # Return (n_steps, n_dims) for single trajectory
            y_result = solution.ys[0]
        else:
            # Transpose from (batch, n_steps, n_dims) to (n_steps, batch, n_dims)
            y_result = solution.ys.transpose(0, 1)

        return t_eval, y_result
