"""AIO State Space scatter plot page with trajectory inspection."""

from collections.abc import Sequence
from typing import Any, cast

import dash_mantine_components as dmc  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]
from dash import (
    MATCH,
    Input,
    NoUpdate,
    Output,
    State,
    callback,  # pyright: ignore[reportUnknownVariableType]
    dcc,
    html,
    no_update,
)

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.plotters.interactive_plotter.bse_base_page_aio import BseBasePageAIO
from pybasin.plotters.interactive_plotter.ids_aio import aio_id
from pybasin.plotters.interactive_plotter.trajectory_modal_aio import TrajectoryModalAIO
from pybasin.plotters.interactive_plotter.utils import get_color
from pybasin.plotters.types import StateSpaceOptions


class StateSpaceAIO(BseBasePageAIO):
    """
    AIO component for state space scatter plot with trajectory inspection.

    Embeds TrajectoryModalAIO for click-to-inspect functionality.
    Uses pattern-matching IDs for instance isolation.
    """

    def __init__(
        self,
        bse: BasinStabilityEstimator,
        aio_id: str,
        state_labels: dict[int, str] | None = None,
        options: StateSpaceOptions | None = None,
    ):
        """
        Initialize state space AIO component.

        Args:
            bse: Basin stability estimator with computed results
            aio_id: Unique identifier for this component instance
            state_labels: Optional mapping of state indices to labels
            options: State space plot configuration options
        """
        super().__init__(bse, aio_id, state_labels)
        self.options = options or StateSpaceOptions()
        StateSpaceAIO._instances[aio_id] = self
        self.trajectory_modal = TrajectoryModalAIO(bse, aio_id, state_labels)

    def render(self) -> html.Div:
        """Render complete page layout with controls, plot, and modal."""
        state_options = self.get_state_options()
        select_data = cast(Sequence[str], state_options)
        n_states = self.get_n_states()

        x_var = self.options.x_var
        y_var = self.options.y_var

        return html.Div(
            [
                dmc.Grid(
                    [
                        dmc.GridCol(
                            [
                                dmc.Flex(
                                    [
                                        dmc.Select(
                                            id=aio_id("StateSpace", self.aio_id, "x-select"),
                                            label="X Axis",
                                            data=select_data,
                                            value=str(x_var),
                                        ),
                                        dmc.Select(
                                            id=aio_id("StateSpace", self.aio_id, "y-select"),
                                            label="Y Axis",
                                            data=select_data,
                                            value=str(y_var) if n_states > 1 else "0",
                                        ),
                                    ],
                                    direction={"base": "row", "md": "column"},
                                    gap="md",
                                    wrap="wrap",
                                    style={
                                        "padding": "16px 8px 16px 16px",
                                        "@media (min-width: 992px)": {"padding": "16px"},
                                    },
                                ),
                            ],
                            span={"base": 12, "md": 2},
                            style={"borderRight": "1px solid #373A40"},
                        ),
                        dmc.GridCol(
                            [
                                dcc.Graph(
                                    id=aio_id("StateSpace", self.aio_id, "plot"),
                                    figure=self.build_figure(x_var=x_var, y_var=y_var),
                                    style={
                                        "width": "100%",
                                        "aspectRatio": "1 / 1",
                                    },
                                    config={
                                        "displayModeBar": True,
                                        "scrollZoom": True,
                                    },
                                ),
                            ],
                            span={"base": 12, "md": 10},
                        ),
                    ],
                ),
                self.trajectory_modal.render(),
            ]
        )

    def build_figure(self, x_var: int = 0, y_var: int = 1) -> go.Figure:
        """Build state space scatter plot."""
        fig = go.Figure()

        if self.bse.y0 is None or self.bse.solution is None:
            fig.add_annotation(  # pyright: ignore[reportUnknownMemberType]
                text="No data available. Run Basin Stability Estimation first.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 14},
            )
            return fig

        y0_np = self.bse.y0.cpu().numpy()
        labels = None
        if self.bse.solution.labels is not None and len(self.bse.solution.labels) > 0:
            labels = np.array(self.bse.solution.labels)

        if labels is None:
            return fig

        unique_labels = np.unique(labels)

        for i, label in enumerate(unique_labels):
            mask = labels == label
            x_data = y0_np[mask, x_var]
            y_data = y0_np[mask, y_var]

            scatter_constructor = go.Scattergl if len(x_data) > 10000 else go.Scatter
            fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
                scatter_constructor(
                    x=x_data,
                    y=y_data,
                    mode="markers",
                    name=str(label),
                    marker={
                        "size": 4,
                        "color": get_color(i),
                        "opacity": 0.7,
                    },
                    hovertemplate=(
                        f"<b>{label}</b><br>"
                        + f"{self.get_state_label(x_var)}: %{{x:.4f}}<br>"
                        + f"{self.get_state_label(y_var)}: %{{y:.4f}}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
            title="State Space",
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


@callback(
    Output(aio_id("StateSpace", MATCH, "plot"), "figure"),
    [
        Input(aio_id("StateSpace", MATCH, "x-select"), "value"),
        Input(aio_id("StateSpace", MATCH, "y-select"), "value"),
    ],
    State(aio_id("StateSpace", MATCH, "plot"), "id"),
    prevent_initial_call=True,
)
def update_state_space_figure_aio(
    x_var: str,
    y_var: str,
    plot_id: dict[str, Any],
) -> go.Figure:
    """Update figure when axis selection changes."""
    instance_id = plot_id["aio_id"]
    instance = StateSpaceAIO.get_instance(instance_id)
    if instance is None or not isinstance(instance, StateSpaceAIO):
        return go.Figure()

    return instance.build_figure(x_var=int(x_var), y_var=int(y_var))


@callback(
    [
        Output(aio_id("TrajectoryModal", MATCH, "modal"), "opened", allow_duplicate=True),
        Output(aio_id("TrajectoryModal", MATCH, "modal"), "title", allow_duplicate=True),
        Output(aio_id("TrajectoryModal", MATCH, "info"), "children", allow_duplicate=True),
        Output(aio_id("TrajectoryModal", MATCH, "sample-store"), "data", allow_duplicate=True),
    ],
    Input(aio_id("StateSpace", MATCH, "plot"), "clickData"),
    [
        State(aio_id("StateSpace", MATCH, "x-select"), "value"),
        State(aio_id("StateSpace", MATCH, "y-select"), "value"),
        State(aio_id("StateSpace", MATCH, "plot"), "id"),
    ],
    prevent_initial_call=True,
)
def open_trajectory_modal_from_state_space_aio(
    click_data: dict[str, Any] | None,
    x_var: str,
    y_var: str,
    plot_id: dict[str, Any],
) -> tuple[
    bool | NoUpdate,
    str | NoUpdate,
    str | NoUpdate,
    dict[str, Any] | None | NoUpdate,
]:
    """Open trajectory modal when a point is clicked."""
    if click_data is None:
        return no_update, no_update, no_update, no_update

    instance_id = plot_id["aio_id"]
    instance = StateSpaceAIO.get_instance(instance_id)
    if instance is None:
        return no_update, no_update, no_update, no_update

    try:
        point = click_data["points"][0]
        sample_idx = point.get("pointIndex")
        if sample_idx is None:
            return no_update, no_update, no_update, no_update

        state_labels = instance.state_labels or {}
        x_label = state_labels.get(int(x_var), f"State {x_var}")
        y_label = state_labels.get(int(y_var), f"State {y_var}")

        x_value = point.get("x")
        y_value = point.get("y")

        title = f"Trajectory {sample_idx}"
        info = f"Clicked on {x_label} = {x_value:.4f}, {y_label} = {y_value:.4f}"

        sample_data: dict[str, Any] = {
            "sample_idx": sample_idx,
            "x_label": x_label,
            "y_label": y_label,
            "x_value": x_value,
            "y_value": y_value,
        }

        return True, title, info, sample_data

    except (KeyError, IndexError, TypeError):
        return no_update, no_update, no_update, no_update
