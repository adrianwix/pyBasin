import logging
from abc import ABC, abstractmethod
from typing import Any, cast

import torch

from pybasin.cache_manager import CacheManager
from pybasin.ode_system import ODESystem
from pybasin.protocols import ODESystemProtocol, SolverProtocol
from pybasin.utils import DisplayNameMixin, resolve_folder

logger = logging.getLogger(__name__)


class Solver(SolverProtocol, DisplayNameMixin, ABC):
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
        n_steps: int | None = None,
        device: str | None = None,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        use_cache: bool = True,
    ):
        """
        Initialize the solver with integration parameters.

        :param time_span: Tuple (t_start, t_end) defining the integration interval.
        :param n_steps: Number of evaluation points. If None, defaults to 1000.
        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        :param rtol: Relative tolerance for adaptive stepping (default: 1e-8).
        :param atol: Absolute tolerance for adaptive stepping (default: 1e-6).
        :param use_cache: Whether to use caching for integration results (default: True).
        """
        self.time_span = time_span
        self.use_cache = use_cache
        self.rtol = rtol
        self.atol = atol

        self.n_steps = n_steps if n_steps is not None else 1000
        self._set_device(device)

        # Only create cache manager if caching is enabled
        # This avoids resolve_folder issues when called from threads
        self._cache_manager: CacheManager | None = None
        self._cache_dir: str | None = None
        if use_cache:
            self._cache_dir = resolve_folder("cache")
            self._cache_manager = CacheManager(self._cache_dir)

    def _set_device(self, device: str | None) -> None:
        """
        Set the device for tensor operations with auto-detection and normalization.

        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        # Store original device string for clone()
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
    def clone(
        self,
        *,
        device: str | None = None,
        n_steps_factor: int = 1,
        use_cache: bool | None = None,
    ) -> "Solver":
        """
        Create a copy of this solver, optionally overriding device, resolution, or caching.

        :param device: Target device ('cpu', 'cuda'). If None, keeps the current device.
        :param n_steps_factor: Multiply the number of evaluation points by this factor.
        :param use_cache: Override caching. If None, keeps the current setting.
        :return: New solver instance.
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
            logger.warning(
                "  Warning: y0 is on device %s but solver expects %s", y0.device, self.device
            )
        if y0.dtype != torch.float32:
            logger.warning("  Warning: y0 has dtype %s but solver expects torch.float32", y0.dtype)

        return t_eval, y0

    def integrate(
        self, ode_system: ODESystemProtocol, y0: torch.Tensor
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
                logger.debug("[%s] Loaded result from cache", self.__class__.__name__)
                return cached_result

        # Compute integration if not cached or cache disabled
        if self.use_cache:
            logger.debug(
                "[%s] Cache miss - integrating on %s...", self.__class__.__name__, self.device
            )
        else:
            logger.debug(
                "[%s] Cache disabled - integrating on %s...", self.__class__.__name__, self.device
            )
        # Cast to concrete type for internal implementation
        ode_system_concrete = cast(ODESystem[Any], ode_system)
        t_result, y_result = self._integrate(ode_system_concrete, y0, t_eval)
        logger.debug("[%s] Integration complete", self.__class__.__name__)

        # Save to cache if enabled
        if self.use_cache and cache_key is not None and self._cache_manager is not None:
            self._cache_manager.save(cache_key, t_result, y_result)

        return t_result, y_result

    def _get_cache_config(self) -> dict[str, Any]:
        """
        Get solver-specific configuration for cache key.
        Subclasses should override this to include additional relevant parameters.

        :return: Dictionary of configuration parameters that affect integration results.
        """
        return {"rtol": self.rtol, "atol": self.atol}

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
