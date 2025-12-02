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

WEBGL_THRESHOLD = 5000


def get_color(idx: int) -> str:
    """Get color from palette by index."""
    return COLORS[idx % len(COLORS)]


def use_webgl(y0: torch.Tensor | None) -> bool:
    """Determine if WebGL should be used based on dataset size."""
    if y0 is None:
        return False
    return len(y0) > WEBGL_THRESHOLD
