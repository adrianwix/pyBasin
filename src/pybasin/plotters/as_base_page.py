# pyright: basic
"""Base class for Adaptive Study plotter pages."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from dash import html

if TYPE_CHECKING:
    from pybasin.as_basin_stability_estimator import ASBasinStabilityEstimator


class ASBasePage(ABC):
    """Abstract base class for Adaptive Study plotter page modules.

    Each page encapsulates:
    - Complete layout generation (controls + figure)
    - Figure building logic
    - Module-level @callback registration for interactivity
    """

    def __init__(
        self,
        as_bse: "ASBasinStabilityEstimator",
        state_labels: dict[int, str] | None = None,
    ):
        self.as_bse = as_bse
        self.state_labels = state_labels or {}

    def get_state_label(self, idx: int) -> str:
        """Get label for a state variable."""
        return self.state_labels.get(idx, f"State {idx}")

    def get_parameter_name_short(self) -> str:
        """Get short display name of the parameter being studied."""
        param_name = self.as_bse.as_params["adaptative_parameter_name"]
        return param_name.split(".")[-1]

    def get_n_parameters(self) -> int:
        """Get number of parameter values in the study."""
        return len(self.as_bse.parameter_values)

    def get_sampler_name(self) -> str:
        """Get display name of the sampler."""
        sampler = self.as_bse.sampler
        display_name = getattr(sampler, "display_name", None)
        return display_name or sampler.__class__.__name__

    def get_solver_name(self) -> str:
        """Get display name of the solver."""
        solver = self.as_bse.solver
        display_name = getattr(solver, "display_name", None)
        return display_name or solver.__class__.__name__

    def get_classifier_name(self) -> str:
        """Get display name of the cluster classifier."""
        classifier = self.as_bse.cluster_classifier
        display_name = getattr(classifier, "display_name", None)
        return display_name or classifier.__class__.__name__

    def get_feature_extractor_name(self) -> str:
        """Get display name of the feature extractor."""
        fe = self.as_bse.feature_extractor
        display_name = getattr(fe, "display_name", None)
        return display_name or fe.__class__.__name__

    def get_n_states(self) -> int:
        """Get number of state variables."""
        return self.as_bse.sampler.state_dim

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
