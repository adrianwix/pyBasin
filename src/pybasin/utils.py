import inspect
import os
import time
from datetime import datetime
from json import JSONEncoder

import numpy as np
import torch

from pybasin.solution import Solution


def time_execution(script_name, func, *args, **kwargs):
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
    pybasin package directory.
    """
    # Get the absolute directory of the current (pybasin) module.
    library_dir = os.path.abspath(os.path.dirname(__file__))

    for frame in inspect.stack():
        caller_file = frame.frame.f_globals.get("__file__")
        if caller_file:
            abs_caller_file = os.path.abspath(caller_file)
            # Check if the caller file is outside of the pybasin module directory.
            if not abs_caller_file.startswith(library_dir):
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
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, Solution):
            return {
                "initial_condition": obj.initial_condition.tolist(),
                "time": obj.time.tolist(),
                "trajectory": obj.trajectory.tolist(),
                "features": obj.features.tolist() if obj.features is not None else None,
                "label": obj.label,
            }
        return super().default(obj)


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
