# pyright: basic
"""Base class for modular plotter pages."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import html

if TYPE_CHECKING:
    from pybasin.basin_stability_estimator import BasinStabilityEstimator


class BasePage(ABC):
    """Abstract base class for plotter page modules.

    Each page encapsulates:
    - Complete layout generation (controls + figure)
    - Figure building logic
    - Module-level @callback registration for interactivity
    """

    def __init__(
        self,
        bse: "BasinStabilityEstimator",
        state_labels: dict[int, str] | None = None,
    ):
        self.bse = bse
        self.state_labels = state_labels or {}

    def get_state_label(self, idx: int) -> str:
        """Get label for a state variable."""
        return self.state_labels.get(idx, f"State {idx}")

    def get_sampler_name(self) -> str:
        """Get display name of the sampler."""
        sampler = self.bse.sampler
        display_name = getattr(sampler, "display_name", None)
        return display_name or sampler.__class__.__name__

    def get_solver_name(self) -> str:
        """Get display name of the solver."""
        solver = self.bse.solver
        display_name = getattr(solver, "display_name", None)
        return display_name or solver.__class__.__name__

    def get_classifier_name(self) -> str:
        """Get display name of the cluster classifier."""
        classifier = self.bse.cluster_classifier
        display_name = getattr(classifier, "display_name", None)
        return display_name or classifier.__class__.__name__

    def get_feature_extractor_name(self) -> str:
        """Get display name of the feature extractor."""
        fe = self.bse.feature_extractor
        display_name = getattr(fe, "display_name", None)
        return display_name or fe.__class__.__name__

    def get_n_states(self) -> int:
        """Get number of state variables."""
        if self.bse.y0 is None:
            return 0
        return self.bse.y0.shape[1]

    def get_state_options(self) -> list[dict[str, str]]:
        """Get dropdown options for state variable selection."""
        return [
            {"value": str(i), "label": self.get_state_label(i)} for i in range(self.get_n_states())
        ]

    @property
    @abstractmethod
    def page_id(self) -> str:
        """Unique identifier for this page (matches IDs prefix)."""
        pass

    @property
    @abstractmethod
    def nav_label(self) -> str:
        """Label shown in navigation."""
        pass

    @property
    @abstractmethod
    def nav_icon(self) -> str:
        """Icon/emoji for navigation."""
        pass

    @abstractmethod
    def build_layout(self) -> html.Div:
        """Build the complete page layout including controls and figure.

        Returns a div containing:
        - Controls panel (if any)
        - Main figure/plot
        - Any additional page-specific components
        """
        pass

    @abstractmethod
    def build_figure(self, **kwargs: object) -> go.Figure:
        """Build the main figure for this page.

        This is used internally by build_layout and by callbacks
        to update the figure when controls change.
        """
        pass

    def build_controls(self) -> dmc.Group | None:
        """Build the controls panel for this page.

        Returns None if no controls are needed (e.g., basin stability bar chart).
        Override in subclasses that need controls.
        """
        return None
