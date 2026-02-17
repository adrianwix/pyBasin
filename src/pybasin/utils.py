import inspect
import os
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime
from json import JSONEncoder
from typing import Any, ParamSpec, TypeVar

import numpy as np
import torch

from pybasin.protocols import FeatureSelectorProtocol
from pybasin.solution import Solution

P = ParamSpec("P")
R = TypeVar("R")

FEATURE_NAME_PATTERN = re.compile(r"^state_(\d+)__(.+)$")


class DisplayNameMixin:
    """Mixin that provides a computed display_name property from the class name."""

    @property
    def display_name(self) -> str:
        """Human-readable name derived from class name (e.g., 'TorchDiffEqSolver' -> 'Torch Diff Eq Solver')."""
        class_name = self.__class__.__name__
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", class_name)
        spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
        return spaced


def parse_feature_name(feature_name: str) -> tuple[int, str] | None:
    """Parse a feature name into state index and feature identifier.

    Feature names follow the convention: state_X__feature_name
    where X is the state dimension index (0-based).

    ```python

    - "state_0__variance" -> (0, "variance")
    - "state_1__mean" -> (1, "mean")
    - "state_0__autocorrelation_periodicity__output_strength" -> (0, "autocorrelation_periodicity__output_strength")
    - "invalid_name" -> None
    ```

    :param feature_name: Feature name string to parse.
    :return: Tuple of (state_index, feature_identifier) if valid, None if invalid.
    """
    match = FEATURE_NAME_PATTERN.match(feature_name)
    if match:
        state_idx = int(match.group(1))
        feature_id = match.group(2)
        return (state_idx, feature_id)
    return None


def validate_feature_names(feature_names: list[str]) -> tuple[bool, list[str]]:
    """Validate that all feature names follow the naming convention.

    Feature names must follow: state_X__feature_name
    where X is a non-negative integer.

    :param feature_names: List of feature names to validate.
    :return: Tuple of (all_valid, invalid_names) where all_valid is True if all names are valid,
        and invalid_names is a list of names that don't match the convention.
    """
    invalid_names: list[str] = []
    for name in feature_names:
        if parse_feature_name(name) is None:
            invalid_names.append(name)
    return (len(invalid_names) == 0, invalid_names)


def get_feature_indices_by_base_name(feature_names: list[str], base_feature: str) -> list[int]:
    """Get column indices for features matching a base feature name.

    Finds all features that have the given base name (after the state_X__ prefix).
    Useful for predictors that need to access specific features across multiple states.

    ```python
    Given feature_names = ["state_0__variance", "state_1__variance", "state_0__mean"]:
    - get_feature_indices_by_base_name(feature_names, "variance") -> [0, 1]
    - get_feature_indices_by_base_name(feature_names, "mean") -> [2]
    ```

    :param feature_names: List of feature names.
    :param base_feature: Base feature name to search for (e.g., "variance", "mean").
    :return: List of column indices where the base feature appears.
    """
    indices: list[int] = []
    for idx, name in enumerate(feature_names):
        parsed = parse_feature_name(name)
        if parsed is not None:
            _, feature_id = parsed
            if feature_id == base_feature or feature_id.startswith(base_feature + "__"):
                indices.append(idx)
    return indices


def time_execution(script_name: str, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    start_time = time.time()  # Record the start time
    result = func(*args, **kwargs)  # Execute the function
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time

    # Get the current time and date in a human-readable format
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Write the elapsed time and current time to a file
    with open("execution_time.txt", "a") as f:
        f.write(f"{current_time} - {script_name}: {elapsed_time} seconds\n")

    return result


def generate_filename(name: str, file_extension: str):
    """
    Generates a unique filename using either a timestamp or a UUID.
    """
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{date}_{name}.{file_extension}"


_PROJECT_ROOT_MARKERS: tuple[str, ...] = ("pyproject.toml", ".git")


def find_project_root(start: str | None = None) -> str:
    """
    Walk up from *start* (default: cwd) until a marker file is found.

    Markers checked: ``pyproject.toml``, ``.git``.

    :param start: Directory to start searching from. Defaults to ``os.getcwd()``.
    :return: Absolute path to the project root directory.
    :raises FileNotFoundError: If no marker is found before reaching the filesystem root.
    """
    current = os.path.abspath(start or os.getcwd())
    while True:
        if any(os.path.exists(os.path.join(current, m)) for m in _PROJECT_ROOT_MARKERS):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError(
                f"Could not find project root (looked for {_PROJECT_ROOT_MARKERS})"
            )
        current = parent


def resolve_cache_dir(cache_dir: str) -> str:
    """
    Resolve a cache directory path and ensure it exists.

    Relative paths are resolved from the project root (found via marker-file detection).
    Absolute paths are used as-is.

    :param cache_dir: Relative or absolute path to the cache directory.
    :return: Absolute path to the cache directory (created if needed).
    """
    if os.path.isabs(cache_dir):
        full_path = cache_dir
    else:
        full_path = os.path.join(find_project_root(), cache_dir)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def _get_caller_dir() -> str:
    """
    Inspects the call stack to determine the directory of the calling script
    that is outside the pybasin module. This implementation iterates over the
    stack frames and returns the first frame whose __file__ is not within the
    pybasin package directory or the Python standard library.
    """
    # Get the absolute directory of the current (pybasin) module.
    library_dir = os.path.abspath(os.path.dirname(__file__))

    # Get standard library paths to exclude them
    stdlib_paths = {os.path.dirname(os.__file__)}
    if hasattr(sys, "base_prefix"):
        stdlib_paths.add(sys.base_prefix)
    if hasattr(sys, "prefix"):
        stdlib_paths.add(sys.prefix)

    for frame in inspect.stack():
        caller_file = frame.frame.f_globals.get("__file__")
        if caller_file:
            abs_caller_file = os.path.abspath(caller_file)
            if not abs_caller_file.startswith(library_dir):
                is_stdlib = any(
                    abs_caller_file.startswith(stdlib_path) for stdlib_path in stdlib_paths
                )
                if not is_stdlib:
                    return os.path.dirname(abs_caller_file)

    return os.getcwd()


def resolve_folder(save_to: str) -> str:
    """
    Resolves the folder path relative to the caller's directory and ensures it exists.
    """
    base_dir = _get_caller_dir()
    full_folder = os.path.join(base_dir, save_to)
    os.makedirs(full_folder, exist_ok=True)
    return full_folder


class NumpyEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:  # type: ignore[override]
        if isinstance(o, np.ndarray):
            return o.tolist()  # type: ignore[return-value]
        if isinstance(o, np.integer):
            return int(o)  # type: ignore[arg-type]
        if isinstance(o, np.floating):
            return float(o)  # type: ignore[arg-type]
        if isinstance(o, Solution):
            return {
                "initial_condition": o.initial_condition.tolist(),  # type: ignore[misc]
                "time": o.time.tolist(),  # type: ignore[misc]
                "y": o.y.tolist(),  # type: ignore[misc]
                "features": o.features.tolist() if o.features is not None else None,  # type: ignore[misc]
                "labels": o.labels.tolist() if o.labels is not None else None,  # type: ignore[misc]
            }
        return super().default(o)


def get_feature_names(selector: FeatureSelectorProtocol, original_names: list[str]) -> list[str]:
    """Get feature names after applying a sklearn selector/transformer.

    :param selector: Fitted feature selector satisfying :class:`FeatureSelectorProtocol`.
    :param original_names: List of original feature names before filtering.
    :return: List of feature names that passed the selector's filter.
    """
    mask = selector.get_support(indices=False)
    return [name for name, keep in zip(original_names, mask, strict=True) if keep]


def extract_amplitudes(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Extract amplitudes by taking the maximum absolute value along the time dimension.

    :param t: Time points tensor (shape: (N,))
    :param y: Input tensor (shape: (N, B, S)) where:
        N = number of time steps
        B = batch size
        S = number of states
    :return: Tensor containing maximum absolute values (shape: (B, S))
    """
    # Take absolute value and maximum along time dimension (dim=0)
    amps = torch.max(torch.abs(y), dim=0)[0]
    return amps
