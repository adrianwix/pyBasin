# pyright: basic
"""State space scatter plot page with trajectory inspection."""

from collections.abc import Sequence
from typing import Any, cast

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import Input, NoUpdate, Output, State, callback, dcc, html, no_update

from pybasin.plotters.base_page import BasePage
from pybasin.plotters.ids import IDs
from pybasin.plotters.trajectory_modal import SELECTED_SAMPLE_DATA, TrajectoryModal
from pybasin.plotters.types import StateSpaceOptions
from pybasin.plotters.utils import get_color, use_webgl


class StateSpacePage(BasePage):
    """Scatter plot of initial conditions colored by attractor label.

    Supports click-to-inspect functionality for viewing individual trajectories.
    """

    # Component IDs using centralized registry
    X_SELECT = IDs.id(IDs.STATE_SPACE, "x-select")
    Y_SELECT = IDs.id(IDs.STATE_SPACE, "y-select")
    PLOT = IDs.id(IDs.STATE_SPACE, "plot")
    CONTROLS = IDs.id(IDs.STATE_SPACE, "controls")

    def __init__(
        self,
        bse: object,
        state_labels: dict[int, str] | None = None,
        options: StateSpaceOptions | None = None,
        id_suffix: str = "",
    ):
        super().__init__(bse, state_labels)  # type: ignore[arg-type]
        self.options = options or StateSpaceOptions()
        self.id_suffix = id_suffix

        # Override class-level IDs with instance-specific ones
        if id_suffix:
            self.X_SELECT = f"{IDs.id(IDs.STATE_SPACE, 'x-select')}-{id_suffix}"
            self.Y_SELECT = f"{IDs.id(IDs.STATE_SPACE, 'y-select')}-{id_suffix}"
            self.PLOT = f"{IDs.id(IDs.STATE_SPACE, 'plot')}-{id_suffix}"
            self.CONTROLS = f"{IDs.id(IDs.STATE_SPACE, 'controls')}-{id_suffix}"

    @property
    def page_id(self) -> str:
        return IDs.STATE_SPACE

    @property
    def nav_label(self) -> str:
        return "State Space"

    @property
    def nav_icon(self) -> str:
        return "🎯"

    def get_time_bounds(self) -> tuple[float, float]:
        """Get the time bounds from the solution."""
        if self.bse.solution is None:
            return 0.0, 1.0
        t = self.bse.solution.time.cpu().numpy()
        return float(t[0]), float(t[-1])

    def get_default_time_span(self) -> tuple[float, float]:
        """Get default time span (0 to 15% of total time)."""
        t_min, t_max = self.get_time_bounds()
        return t_min, t_min + (t_max - t_min) * 0.15

    def build_layout(self) -> html.Div:
        """Build complete page layout with controls and plot.

        :return: Div containing controls and scatter plot.
        """
        state_options = self.get_state_options()
        select_data = cast(Sequence[str], state_options)
        n_states = self.get_n_states()

        x_var = self.options.x_var
        y_var = self.options.y_var

        return html.Div(
            [
                # Controls panel
                dmc.Paper(
                    [
                        dmc.Group(
                            [
                                dmc.Select(
                                    id=self.X_SELECT,
                                    label="X Axis",
                                    data=select_data,
                                    value=str(x_var),
                                    w=150,
                                ),
                                dmc.Select(
                                    id=self.Y_SELECT,
                                    label="Y Axis",
                                    data=select_data,
                                    value=str(y_var) if n_states > 1 else "0",
                                    w=150,
                                ),
                            ],
                            gap="md",
                        ),
                    ],
                    p="md",
                    mb="md",
                    withBorder=True,
                    id=self.CONTROLS,
                ),
                # Main plot
                dmc.Paper(
                    [
                        dcc.Graph(
                            id=self.PLOT,
                            figure=self.build_figure(x_var=x_var, y_var=y_var),
                            style={
                                "height": "70vh",
                                "aspectRatio": "1 / 1",
                                "maxWidth": "70vh",
                                "margin": "0 auto",
                            },
                            config={
                                "displayModeBar": True,
                                "scrollZoom": True,
                            },
                        ),
                    ],
                    p="md",
                    withBorder=True,
                ),
            ]
        )

    def build_controls(self) -> dmc.Group | None:
        state_options = self.get_state_options()
        select_data = cast(Sequence[str], state_options)
        n_states = self.get_n_states()

        return dmc.Group(
            [
                dmc.Select(
                    id=self.X_SELECT,
                    label="X Axis",
                    data=select_data,
                    value="0",
                    w=150,
                ),
                dmc.Select(
                    id=self.Y_SELECT,
                    label="Y Axis",
                    data=select_data,
                    value="1" if n_states > 1 else "0",
                    w=150,
                ),
            ],
            gap="md",
        )

    def build_figure(self, x_var: int = 0, y_var: int = 1, **kwargs: object) -> go.Figure:
        if self.bse.y0 is None or self.bse.solution is None:
            return go.Figure()

        initial_conditions = self.bse.y0.cpu().numpy()
        labels = np.array(self.bse.solution.labels)
        unique_labels = np.unique(labels)

        fig = go.Figure()
        scatter_type = go.Scattergl if use_webgl(self.bse.y0) else go.Scatter

        for i, label in enumerate(unique_labels):
            idx = np.where(labels == label)[0]
            fig.add_trace(
                scatter_type(
                    x=initial_conditions[idx, x_var],
                    y=initial_conditions[idx, y_var],
                    mode="markers",
                    marker={"size": 4, "color": get_color(i), "opacity": 0.6},
                    name=str(label),
                    customdata=idx.reshape(-1, 1),
                    hovertemplate=(
                        f"{self.get_state_label(x_var)}: %{{x:.4f}}<br>"
                        f"{self.get_state_label(y_var)}: %{{y:.4f}}<br>"
                        "Index: %{customdata[0]}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title="Initial Conditions in State Space (click to inspect)",
            xaxis_title=self.get_state_label(x_var),
            yaxis_title=self.get_state_label(y_var),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
            },
            hovermode="closest",
        )

        return fig


# Global instance placeholder - will be set by InteractivePlotter
_page_instance: StateSpacePage | None = None


def set_page_instance(page: StateSpacePage) -> None:
    """Set the global page instance for callbacks."""
    global _page_instance
    _page_instance = page


@callback(
    Output(StateSpacePage.PLOT, "figure"),
    [
        Input(StateSpacePage.X_SELECT, "value"),
        Input(StateSpacePage.Y_SELECT, "value"),
    ],
    prevent_initial_call=True,
)
def update_state_space_figure(x_var: str, y_var: str) -> go.Figure:
    """Update figure when axis selection changes."""
    if _page_instance is None:
        return go.Figure()
    return _page_instance.build_figure(x_var=int(x_var), y_var=int(y_var))


@callback(
    [
        Output(TrajectoryModal.MODAL, "opened", allow_duplicate=True),
        Output(TrajectoryModal.MODAL, "title", allow_duplicate=True),
        Output(TrajectoryModal.MODAL_INFO, "children", allow_duplicate=True),
        Output(SELECTED_SAMPLE_DATA, "data", allow_duplicate=True),
    ],
    Input(StateSpacePage.PLOT, "clickData"),
    [
        State(StateSpacePage.X_SELECT, "value"),
        State(StateSpacePage.Y_SELECT, "value"),
    ],
    prevent_initial_call=True,
)
def open_trajectory_modal_from_state_space(
    click_data: dict[str, Any] | None,
    x_var: str,
    y_var: str,
) -> tuple[
    bool | NoUpdate,
    str | NoUpdate,
    str | NoUpdate,
    dict[str, Any] | None | NoUpdate,
]:
    """Open trajectory modal when a point is clicked."""
    if _page_instance is None or click_data is None:
        return no_update, no_update, no_update, no_update

    bse = _page_instance.bse
    if bse.solution is None or bse.solution.labels is None:
        return no_update, no_update, no_update, no_update

    try:
        point = click_data["points"][0]
        curve_number = point["curveNumber"]
        point_index = point["pointIndex"]

        labels = np.array(bse.solution.labels)
        unique_labels = np.unique(labels)

        if curve_number >= len(unique_labels):
            return no_update, no_update, no_update, no_update

        target_label = unique_labels[curve_number]
        label_indices = np.where(labels == target_label)[0]

        if point_index >= len(label_indices):
            return no_update, no_update, no_update, no_update

        sample_idx = int(label_indices[point_index])
        label = bse.solution.labels[sample_idx]

        x_coord = point.get("x", 0)
        y_coord = point.get("y", 0)
        x_label = _page_instance.get_state_label(int(x_var))
        y_label = _page_instance.get_state_label(int(y_var))

        title = f"Trajectory: Sample {sample_idx} (Label: {label})"
        info = f"Initial Conditions: {x_label} = {x_coord:.4f}, {y_label} = {y_coord:.4f}"

        return (
            True,
            title,
            info,
            {"sample_idx": sample_idx, "label": str(label)},
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return no_update, no_update, no_update, no_update
