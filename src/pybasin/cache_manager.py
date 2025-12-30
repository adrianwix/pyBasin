import hashlib
import logging
import os
import pickle
import shutil
from typing import Any

import torch

from pybasin.protocols import ODESystemProtocol

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages persistent caching of integration results."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def build_key(
        self,
        solver_name: str,
        ode_system: ODESystemProtocol,
        y0: torch.Tensor,
        t_eval: torch.Tensor,
        solver_config: dict[str, Any] | None = None,
    ) -> str:
        """Build a unique cache key based on solver type, configuration, ODE system, and initial conditions.

        :param solver_name: Name of the solver class.
        :param ode_system: The ODE system being solved.
        :param y0: Initial conditions.
        :param t_eval: Time evaluation points.
        :param solver_config: Dictionary of solver-specific parameters (rtol, atol, method, etc.).
        """
        # Automatically include ODE system parameters in the cache key
        # This ensures different parameter values produce different cache keys
        params_tuple = (
            tuple(sorted(ode_system.params.items())) if hasattr(ode_system, "params") else ()
        )

        key_data = (
            solver_name,
            ode_system.get_str(),
            params_tuple,
            y0.detach().cpu().numpy().tobytes(),
            t_eval.detach().cpu().numpy().tobytes(),
            tuple(sorted(solver_config.items())) if solver_config else (),
        )
        key_bytes = pickle.dumps(key_data)
        return hashlib.md5(key_bytes).hexdigest()

    def load(
        self, cache_key: str, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Load cached result from disk if it exists."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if not os.path.exists(cache_file):
            return None

        try:
            with open(cache_file, "rb") as f:
                t_cached, y_cached = pickle.load(f)
                return t_cached.to(device), y_cached.to(device)
        except EOFError:
            logger.warning("Cache file corrupted. Deleting and recomputing.")
            os.remove(cache_file)
            return None

    def save(self, cache_key: str, t: torch.Tensor, y: torch.Tensor) -> None:
        """Save integration result to disk cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        # Ensure cache directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        # Check available disk space
        usage = shutil.disk_usage(os.path.dirname(cache_file))
        free_gb = usage.free / (1024**3)
        if free_gb < 1:
            logger.warning("Only %.2fGB free space available.", free_gb)

        try:
            # Move to CPU before caching to avoid device issues
            with open(cache_file, "wb") as f:
                pickle.dump((t.cpu(), y.cpu()), f)
        except OSError as e:
            logger.error("Error saving to cache: %s", e)
            raise
