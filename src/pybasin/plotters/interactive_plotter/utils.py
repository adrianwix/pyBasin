# pyright: basic
"""Shared utilities for plotter components."""

import torch

COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def get_color(idx: int) -> str:
    """Get color from palette by index."""
    return COLORS[idx % len(COLORS)]


def tensor_to_float_list(tensor: torch.Tensor) -> list[float]:
    """Convert a torch tensor to a list of floats.

    :param tensor: 1D torch tensor to convert.
    :return: List of float values from the tensor.
    """
    return [float(x) for x in tensor.cpu().tolist()]
