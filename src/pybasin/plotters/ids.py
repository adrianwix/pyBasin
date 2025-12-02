# pyright: basic
"""Centralized ID registry for Dash components.

This module provides a single source of truth for all component IDs used
across the interactive plotter pages, ensuring consistent naming and
avoiding ID collisions.
"""


class IDs:
    """Registry of component ID prefixes and utilities.

    Each page has a unique prefix. Use the `id()` method to generate
    fully qualified component IDs.

    Example:
        >>> IDs.id(IDs.STATE_SPACE, "plot")
        'state-plot'
        >>> IDs.id(IDs.FEATURE_SPACE, "x-select")
        'feature-x-select'
    """

    # Page prefixes - aligned with ViewType from types.py
    BASIN_STABILITY = "bs"
    STATE_SPACE = "state"
    FEATURE_SPACE = "feature"
    PHASE_2D = "phase-2d"
    PHASE_3D = "phase-3d"
    TEMPLATE_TS = "template-ts"

    # Shared component prefixes
    TRAJECTORY_MODAL = "trajectory-modal"

    @staticmethod
    def id(prefix: str, suffix: str) -> str:
        """Generate a fully qualified component ID.

        :param prefix: The page or component prefix (e.g., IDs.STATE_SPACE).
        :param suffix: The component-specific suffix (e.g., "plot", "x-select").
        :return: Combined ID string like "state-plot".
        """
        return f"{prefix}-{suffix}"
