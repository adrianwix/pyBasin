import inspect
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime
from json import JSONEncoder
from typing import Any, ParamSpec, TypeVar

import numpy as np
import torch

from pybasin.solution import Solution

P = ParamSpec("P")
R = TypeVar("R")


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


def _get_caller_dir():
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
            # Check if the caller file is outside of the pybasin module directory
            # and not in the standard library
            if not abs_caller_file.startswith(library_dir):
                # Skip standard library paths
                is_stdlib = any(
                    abs_caller_file.startswith(stdlib_path) for stdlib_path in stdlib_paths
                )
                if not is_stdlib:
                    return os.path.dirname(abs_caller_file)

    # Fallback if __file__ is not found (e.g. interactive shell)
    return os.getcwd()


def resolve_folder(save_to: str):
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


def extract_amplitudes(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Extract amplitudes by taking the maximum absolute value along the time dimension.

    Args:
        t: Time points tensor (shape: (N,))
        y: Input tensor (shape: (N, B, S)) where:
           N = number of time steps
           B = batch size
           S = number of states
        props: Optional properties dictionary

    Returns:
        amps: Tensor containing maximum absolute values (shape: (B, S))
    """
    # Take absolute value and maximum along time dimension (dim=0)
    amps = torch.max(torch.abs(y), dim=0)[0]
    return amps
