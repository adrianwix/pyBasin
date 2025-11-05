"""Utility functions shared across case studies.

This module provides common functionality for:
- Path management
- Result saving
- Plotting configurations
- Data formatting
"""
import os
from pathlib import Path
from typing import Union

# Configure matplotlib to use non-GUI backend for case studies
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_artifacts_dir(subdir: str = "") -> Path:
    """Get the artifacts directory, creating it if necessary.
    
    Args:
        subdir: Optional subdirectory within artifacts (e.g., 'figures', 'results')
    
    Returns:
        Path to the artifacts directory
    """
    artifacts_path = get_project_root() / "artifacts"
    if subdir:
        artifacts_path = artifacts_path / subdir
    artifacts_path.mkdir(parents=True, exist_ok=True)
    return artifacts_path


def get_case_study_name() -> str:
    """Get the name of the current case study from the calling module."""
    import inspect
    frame = inspect.currentframe()
    if frame and frame.f_back:
        caller_file = frame.f_back.f_code.co_filename
        # Extract case study name from path
        path_parts = Path(caller_file).parts
        if 'case_studies' in path_parts:
            idx = path_parts.index('case_studies')
            if idx + 1 < len(path_parts):
                return path_parts[idx + 1]
    return "unknown"


def save_figure(fig, filename: str, subdir: str = "figures", **kwargs):
    """Save a matplotlib figure to the artifacts directory.
    
    Args:
        fig: Matplotlib figure object
        filename: Name of the file (without path)
        subdir: Subdirectory within artifacts
        **kwargs: Additional arguments passed to fig.savefig()
    """
    filepath = get_artifacts_dir(subdir) / filename
    fig.savefig(filepath, **kwargs)
    print(f"Figure saved to: {filepath}")
