# pyright: basic
"""Type definitions for interactive plotter options."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TemplateSelectionOptions:
    """Base options for template selection (include or exclude, not both)."""

    include_templates: list[str] | None = None
    exclude_templates: list[str] | None = None

    def __post_init__(self) -> None:
        if self.include_templates is not None and self.exclude_templates is not None:
            raise ValueError(
                "Cannot specify both include_templates and exclude_templates. "
                "Use include_templates to show only specific templates, "
                "or exclude_templates to hide specific templates."
            )

    def filter_templates(self, all_labels: list[str]) -> list[str]:
        """Filter template labels based on include/exclude settings.

        :param all_labels: All available template labels.
        :return: Filtered list of template labels to display.
        """
        if self.include_templates is not None:
            return [label for label in all_labels if label in self.include_templates]
        elif self.exclude_templates is not None:
            return [label for label in all_labels if label not in self.exclude_templates]
        return all_labels


@dataclass
class StateSpaceOptions:
    """Options for the State Space page."""

    x_var: int = 0
    y_var: int = 1
    time_range_percent: float = 0.15


@dataclass
class FeatureSpaceOptions:
    """Options for the Feature Space page."""

    x_feature: int = 0
    y_feature: int | None = 1
    use_filtered: bool = True
    include_labels: list[str] | None = None
    exclude_labels: list[str] | None = None

    def __post_init__(self) -> None:
        if self.include_labels is not None and self.exclude_labels is not None:
            raise ValueError(
                "Cannot specify both include_labels and exclude_labels. "
                "Use include_labels to show only specific labels, "
                "or exclude_labels to hide specific labels."
            )

    def filter_labels(self, all_labels: list[str]) -> list[str]:
        """Filter labels based on include/exclude settings.

        :param all_labels: All available labels.
        :return: Filtered list of labels to display.
        """
        if self.include_labels is not None:
            return [label for label in all_labels if label in self.include_labels]
        elif self.exclude_labels is not None:
            return [label for label in all_labels if label not in self.exclude_labels]
        return all_labels


@dataclass
class PhasePlotOptions(TemplateSelectionOptions):
    """Options for the Phase Plot 2D/3D pages."""

    x_var: int = 0
    y_var: int = 1
    z_var: int | None = None  # None = auto-infer from remaining state

    def get_z_var(self, n_states: int) -> int:
        """Get the Z-axis variable index, inferring if not set.

        :param n_states: Total number of state variables.
        :return: Z-axis variable index.
        """
        if self.z_var is not None:
            return self.z_var
        if n_states >= 3:
            # Find the first state not used by X or Y
            used = {self.x_var, self.y_var}
            remaining = [i for i in range(n_states) if i not in used]
            return remaining[0] if remaining else 0
        return 0


@dataclass
class TemplateTimeSeriesOptions(TemplateSelectionOptions):
    """Options for the Template Time Series page."""

    state_var: int = 0
    time_range_percent: float = 0.15


@dataclass
class ParamOverviewOptions:
    """Options for the Parameter Overview page."""

    x_scale: Literal["linear", "log"] = "linear"
    selected_labels: list[str] | None = None


@dataclass
class ParamBifurcationOptions:
    """Options for the Parameter Bifurcation page."""

    selected_dofs: list[int] | None = None


ViewType = Literal[
    "bs",
    "state",
    "feature",
    "phase-2d",
    "phase-3d",
    "template-ts",
    "param-overview",
    "param-bifurcation",
]


@dataclass
class InteractivePlotterOptions:
    """Configuration options for InteractivePlotter defaults.

    Use this to customize the initial state of controls for each visualization
    page. Invalid values (e.g., out-of-bounds indices) will trigger a warning
    and fall back to safe defaults.

    ```python
    options = InteractivePlotterOptions(
        initial_view="phase-2d",
        phase_plot=PhasePlotOptions(x_var=0, y_var=2, exclude_templates=["unbounded"]),
        template_ts=TemplateTimeSeriesOptions(time_range_percent=0.10),
    )
    plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "y"}, options=options)
    ```
    """

    initial_view: ViewType = "bs"
    state_space: StateSpaceOptions = field(default_factory=StateSpaceOptions)
    feature_space: FeatureSpaceOptions = field(default_factory=FeatureSpaceOptions)
    phase_plot: PhasePlotOptions = field(default_factory=PhasePlotOptions)
    template_ts: TemplateTimeSeriesOptions = field(default_factory=TemplateTimeSeriesOptions)
    param_overview: ParamOverviewOptions = field(default_factory=ParamOverviewOptions)
    param_bifurcation: ParamBifurcationOptions = field(default_factory=ParamBifurcationOptions)
