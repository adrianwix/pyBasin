import os
import pickle
import hashlib
import shutil
from abc import ABC, abstractmethod

import numpy as np
import torch
from torchdiffeq import odeint

from pybasin.ODESystem import ODESystem
from pybasin.utils import resolve_folder


class Solver(ABC):
    """Abstract base class for ODE solvers with persistent caching.

    The cache is stored both in-memory and on disk.
    The cache key is built using:
      - The solver class name,
      - The ODE system’s string representation via ode_system.get_str(),
      - The serialized initial conditions (y0),
      - The serialized evaluation time points (t_eval).
    The persistent cache is stored in the folder given by resolve_folder("cache").
    """

    def __init__(self, time_span: tuple[float, float], fs: float, **kwargs):
        """
        Initialize the solver with integration parameters.

        :param time_span: Tuple (t_start, t_end) defining the integration interval.
        :param fs: Sampling frequency (Hz) – number of samples per time unit.
        """
        self.time_span = time_span
        self.fs = fs
        self.n_steps = int((time_span[1] - time_span[0]) * fs) + 1
        self.params = kwargs  # Additional solver parameters
        self._cache_dir = resolve_folder("cache")  # Persistent cache folder

    def _build_cache_key(self, ode_system: ODESystem, y0: torch.Tensor, t_eval: torch.Tensor) -> str:
        """
        Build a unique cache key based on:
          - The solver type,
          - The string representation of the ODE system,
          - The contents of y0 and t_eval.
        """
        key_data = (
            self.__class__.__name__,
            ode_system.get_str(),  # String representation of the ODE system
            y0.detach().cpu().numpy().tobytes(),
            t_eval.detach().cpu().numpy().tobytes()
        )
        key_bytes = pickle.dumps(key_data)
        return hashlib.md5(key_bytes).hexdigest()

    def integrate(self, ode_system: ODESystem, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the ODE system and return the evaluation time points and solution.
        Uses caching to avoid recomputation if the same problem was solved before.

        :param ode_system: An instance of ODESystem.
        :param y0: Initial conditions.
        :return: Tuple (t_eval, y_values) where y_values is the solution.
        """
        t_start, t_end = self.time_span
        t_eval = torch.linspace(
            t_start, t_end, self.n_steps, dtype=torch.float64)

        # Build the unique cache key.
        cache_key = self._build_cache_key(ode_system, y0, t_eval)
        cache_file = os.path.join(self._cache_dir, f"{cache_key}.pkl")

        # Check persistent cache on disk.
        if os.path.exists(cache_file):
            print(
                f"[{self.__class__.__name__}] Loading integration result from persistent cache.")
            try:
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)
                return result
            except EOFError:
                print(
                    "EOFError: The cache file may be corrupted. Deleting it and proceeding without cache.")
                os.remove(cache_file)

        # Compute the integration if not cached.
        print(f"[{self.__class__.__name__}] Cache miss. Integrating...")
        result = self._integrate(ode_system, y0, t_eval)

        # Check available disk space (in GB)
        usage = shutil.disk_usage(os.path.dirname(cache_file))
        free_gb = usage.free / (1024**3)
        if free_gb < 1:  # set a threshold (e.g., 1GB)
            print(f"\nWarning: Only {free_gb:.2f}GB free space available.")

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
        except OSError as e:
            print(f"Error during pickle.dump: {e}")
            raise

        print(
            f"[{self.__class__.__name__}] Integration result cached to file: {cache_file}")
        return result

    @abstractmethod
    def _integrate(self, ode_system: ODESystem, y0: torch.Tensor, t_eval: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _integrate(self, ode_system: ODESystem, y0: torch.Tensor, t_eval: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            y_torch = odeint(
                ode_system,
                y0,
                t_eval,
                rtol=1e-8,
                atol=1e-6
            )
        except RuntimeError as e:
            raise e
        return t_eval, y_torch
