# pyright: basic
"""Shared trajectory inspection modal component.

This modal is used by State Space and Feature Space pages to display
trajectory time series for clicked data points.
"""

from typing import TYPE_CHECKING, Any

import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc

from pybasin.plotters.ids import IDs
from pybasin.plotters.utils import get_color

if TYPE_CHECKING:
    from pybasin.basin_stability_estimator import BasinStabilityEstimator


# Store ID for selected sample data (global)
SELECTED_SAMPLE_DATA = "selected-sample-data"


class TrajectoryModal:
    """Shared modal component for trajectory inspection.

    This modal displays time series trajectories when a user clicks on
    a data point in the State Space or Feature Space scatter plots.
    """

    # Component IDs using the centralized registry
    MODAL = IDs.id(IDs.TRAJECTORY_MODAL, "modal")
    MODAL_TITLE = IDs.id(IDs.TRAJECTORY_MODAL, "title")
    MODAL_INFO = IDs.id(IDs.TRAJECTORY_MODAL, "info")
    STATE_SELECT = IDs.id(IDs.TRAJECTORY_MODAL, "state-select")
    TIME_RANGE = IDs.id(IDs.TRAJECTORY_MODAL, "time-range")
    PLOT = IDs.id(IDs.TRAJECTORY_MODAL, "plot")

    def __init__(
        self,
        bse: "BasinStabilityEstimator",
        state_labels: dict[int, str] | None = None,
    ):
        """Initialize the TrajectoryModal.

        :param bse: BasinStabilityEstimator with computed results.
        :param state_labels: Optional mapping of state indices to labels.
        """
        self.bse = bse
        self.state_labels = state_labels or {}

    def get_state_label(self, idx: int) -> str:
        """Get label for a state variable."""
        return self.state_labels.get(idx, f"State {idx}")

    def get_n_states(self) -> int:
        """Get number of state variables."""
        if self.bse.y0 is None:
            return 0
        return self.bse.y0.shape[1]

    def get_time_bounds(self) -> tuple[float, float]:
        """Get the time bounds from the solution."""
        if self.bse.solution is None:
            return 0.0, 1.0
        t = self.bse.solution.time.cpu().numpy()
        return float(t[0]), float(t[-1])

    def build_layout(
        self,
        default_time_range_percent: float = 0.15,
    ) -> dmc.Modal:
        """Build the modal layout.

        :param default_time_range_percent: Default time range as percentage of total.
        :return: Modal component with all controls and plot.
        """
        n_states = self.get_n_states()
        state_options = [
            {"value": str(i), "label": self.get_state_label(i)} for i in range(n_states)
        ]

        t_min, t_max = self.get_time_bounds()
        default_t_max = t_min + (t_max - t_min) * default_time_range_percent

        return dmc.Modal(
            id=self.MODAL,
            title="Trajectory Viewer",
            size="xl",
            centered=True,
            children=[
                dmc.Text(
                    id=self.MODAL_INFO,
                    size="sm",
                    c="gray",
                    mb="md",
                ),
                dmc.Group(
                    [
                        dmc.Select(
                            id=self.STATE_SELECT,
                            label="State Variable",
                            data=state_options,  # type: ignore[arg-type]
                            value="0",
                            w=150,
                        ),
                    ],
                    gap="md",
                    mb="md",
                ),
                dmc.Text("Time Range", size="sm", fw="normal"),
                dmc.RangeSlider(
                    id=self.TIME_RANGE,
                    min=t_min,
                    max=t_max,
                    value=[t_min, default_t_max],  # type: ignore[arg-type]
                    step=(t_max - t_min) / 100 if t_max > t_min else 0.01,
                    minRange=(t_max - t_min) / 50 if t_max > t_min else 0.01,
                    marks=[
                        {"value": t_min, "label": f"{t_min:.1f}"},
                        {"value": t_max, "label": f"{t_max:.1f}"},
                    ],
                    mb="md",
                ),
                dcc.Graph(
                    id=self.PLOT,
                    figure=_build_empty_figure(),
                    style={"height": "40vh"},
                    config={"displayModeBar": True},
                ),
            ],
        )

    def build_time_series_figure(
        self,
        sample_idx: int | None = None,
        state_var: int = 0,
        time_span: tuple[float, float] | None = None,
    ) -> go.Figure:
        """Build time series plot for a selected sample.

        :param sample_idx: Index of the sample to plot.
        :param state_var: State variable index to display.
        :param time_span: Optional (t_min, t_max) to limit time range.
        :return: Plotly figure with the trajectory.
        """
        if self.bse.solution is None:
            return _build_empty_figure()

        fig = go.Figure()

        if sample_idx is not None:
            t = self.bse.solution.time.cpu().numpy()
            y = self.bse.solution.y.cpu().numpy()

            if time_span is not None:
                mask = (t >= time_span[0]) & (t <= time_span[1])
                t = t[mask]
                y = y[mask]

            trajectory = y[:, sample_idx, state_var]
            label = (
                self.bse.solution.labels[sample_idx]
                if self.bse.solution.labels is not None
                else "Unknown"
            )

            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=trajectory,
                    mode="lines",
                    name=f"Sample {sample_idx} ({label})",
                    line={"color": get_color(0), "width": 2},
                )
            )

            fig.update_layout(title=f"Trajectory: {self.get_state_label(state_var)}")
        else:
            fig.add_annotation(
                text="Click a point in the scatter plot to view its trajectory",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 14},
            )
            fig.update_layout(title="Trajectory Viewer")

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=self.get_state_label(state_var),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        return fig


def _build_empty_figure() -> go.Figure:
    """Build an empty placeholder figure."""
    fig = go.Figure()
    fig.add_annotation(
        text="Click a point in the scatter plot to view its trajectory",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14},
    )
    fig.update_layout(
        title="Trajectory Viewer",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# Global instance placeholder - will be set by InteractivePlotter
_modal_instance: TrajectoryModal | None = None


def set_modal_instance(modal: TrajectoryModal) -> None:
    """Set the global modal instance for callbacks."""
    global _modal_instance  # noqa: PLW0603
    _modal_instance = modal


@callback(
    Output(TrajectoryModal.PLOT, "figure"),
    [
        Input(SELECTED_SAMPLE_DATA, "data"),
        Input(TrajectoryModal.STATE_SELECT, "value"),
        Input(TrajectoryModal.TIME_RANGE, "value"),
    ],
)
def update_trajectory_plot(
    sample_data: dict[str, Any] | None,
    state_var: str,
    time_range: list[float],
) -> go.Figure:
    """Update the trajectory plot when sample or controls change."""
    if _modal_instance is None:
        return _build_empty_figure()

    if sample_data is None:
        return _modal_instance.build_time_series_figure()

    sample_idx = sample_data.get("sample_idx")
    if sample_idx is None:
        return _modal_instance.build_time_series_figure()

    time_span = tuple(time_range) if time_range else None
    return _modal_instance.build_time_series_figure(
        sample_idx=int(sample_idx),
        state_var=int(state_var),
        time_span=time_span,  # type: ignore[arg-type]
    )
