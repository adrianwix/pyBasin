"""AIO Template Time Series page."""

from collections.abc import Sequence
from typing import Any, cast

import dash_mantine_components as dmc  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]
from dash import (
    MATCH,
    Input,
    Output,
    State,
    callback,  # pyright: ignore[reportUnknownVariableType]
    dcc,
    html,
)
from dash.development.base_component import Component

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.cluster_classifier import SupervisedClassifier
from pybasin.plotters.interactive_plotter.bse_base_page_aio import BseBasePageAIO
from pybasin.plotters.interactive_plotter.ids_aio import aio_id
from pybasin.plotters.interactive_plotter.trajectory_cache import TrajectoryCache
from pybasin.plotters.interactive_plotter.utils import get_color
from pybasin.plotters.types import TemplateTimeSeriesOptions


class TemplateTimeSeriesAIO(BseBasePageAIO):
    """
    AIO component for template trajectory time series plots.

    Displays time series for template trajectories from SupervisedClassifier.
    """

    def __init__(
        self,
        bse: BasinStabilityEstimator,
        aio_id: str,
        state_labels: dict[int, str] | None = None,
        options: TemplateTimeSeriesOptions | None = None,
    ):
        """
        Initialize template time series AIO component.

        Args:
            bse: Basin stability estimator with computed results
            aio_id: Unique identifier for this component instance
            state_labels: Optional mapping of state indices to labels
            options: Template time series configuration options
        """
        super().__init__(bse, aio_id, state_labels)
        self.options = options or TemplateTimeSeriesOptions()
        TemplateTimeSeriesAIO._instances[aio_id] = self

    def get_time_bounds(self) -> tuple[float, float]:
        """Get time bounds from integrated solution."""
        t, _ = TrajectoryCache.get_or_integrate(self.bse)
        return float(t[0]), float(t[-1])

    def render(self) -> html.Div:
        """Render complete page layout with controls and time series plot."""
        if not isinstance(self.bse.cluster_classifier, SupervisedClassifier):
            return html.Div(
                dmc.Paper(
                    dmc.Text(
                        "Template time series requires SupervisedClassifier",
                        size="lg",
                        c="gray",
                    ),
                    p="xl",
                    withBorder=True,
                )
            )

        state_options = self.get_state_options()
        all_template_labels = list(self.bse.cluster_classifier.labels)
        selected_templates = self.options.filter_templates(all_template_labels)

        t_min, t_max = self.get_time_bounds()
        default_span = (t_min, t_min + (t_max - t_min) * self.options.time_range_percent)

        state_var = self.options.state_var
        select_data = cast(Sequence[str], state_options)

        controls: list[Component] = [
            dmc.Select(
                id=aio_id("TemplateTimeSeries", self.aio_id, "state-select"),
                label="State Variable",
                data=select_data,
                value=str(state_var),
                w=150,
            ),
            dmc.MultiSelect(
                id=aio_id("TemplateTimeSeries", self.aio_id, "template-select"),
                label="Templates",
                data=all_template_labels,
                value=selected_templates,
                w=300,
            ),
        ]

        return html.Div(
            [
                dmc.Paper(
                    [
                        dmc.Group(controls, gap="md"),
                        dmc.Space(h="md"),
                        dmc.Text("Time Range", size="sm", fw="normal"),
                        dcc.RangeSlider(
                            id=aio_id("TemplateTimeSeries", self.aio_id, "time-slider"),
                            min=t_min,
                            max=t_max,
                            step=(t_max - t_min) / 100,
                            value=[default_span[0], default_span[1]],
                            marks={
                                t_min: {"label": f"{t_min:.1f}"},
                                t_max: {"label": f"{t_max:.1f}"},
                            },
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                    p="md",
                    withBorder=True,
                    mb="md",
                ),
                dmc.Paper(
                    [
                        dcc.Graph(
                            id=aio_id("TemplateTimeSeries", self.aio_id, "plot"),
                            figure=self.build_figure(
                                state_var=state_var,
                                time_span=default_span,
                                selected_templates=selected_templates,
                            ),
                            style={"height": "70vh"},
                            config={
                                "displayModeBar": True,
                                "scrollZoom": True,
                            },
                        ),
                    ],
                    p="md",
                    withBorder=True,
                ),
            ],
            style={"padding": "16px"},
        )

    def build_figure(
        self,
        state_var: int = 0,
        time_span: tuple[float, float] | None = None,
        selected_templates: list[str] | None = None,
    ) -> go.Figure:
        """Build template time series plot."""
        if not isinstance(self.bse.cluster_classifier, SupervisedClassifier):
            fig = go.Figure()
            fig.add_annotation(  # pyright: ignore[reportUnknownMemberType]
                text="Template plot requires SupervisedClassifier",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font={"size": 14},
            )
            return fig

        t, y = TrajectoryCache.get_or_integrate(self.bse)

        if time_span is not None:
            mask = (t >= time_span[0]) & (t <= time_span[1])
            t = t[mask]
            y = y[mask]

        fig = go.Figure()

        # After isinstance check, type is narrowed to SupervisedClassifier
        all_labels = self.bse.cluster_classifier.labels
        if selected_templates is None:
            selected_templates = list(all_labels)

        for i, (label, traj) in enumerate(
            zip(
                all_labels,
                np.transpose(y, (1, 0, 2)),
                strict=True,
            )
        ):
            if label not in selected_templates:
                continue
            fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
                go.Scatter(
                    x=t,
                    y=traj[:, state_var],
                    mode="lines",
                    name=str(label),
                    line={"color": get_color(i), "width": 2},
                )
            )

        fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
            title=f"Template Trajectories - {self.get_state_label(state_var)}",
            xaxis_title="Time",
            yaxis_title=self.get_state_label(state_var),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        return fig


@callback(
    Output(aio_id("TemplateTimeSeries", MATCH, "plot"), "figure"),
    [
        Input(aio_id("TemplateTimeSeries", MATCH, "state-select"), "value"),
        Input(aio_id("TemplateTimeSeries", MATCH, "template-select"), "value"),
        Input(aio_id("TemplateTimeSeries", MATCH, "time-slider"), "value"),
    ],
    State(aio_id("TemplateTimeSeries", MATCH, "plot"), "id"),
    prevent_initial_call=True,
)
def update_template_time_series_figure_aio(
    state_var: str,
    templates: list[str],
    time_range: list[float],
    plot_id: dict[str, Any],
) -> go.Figure:
    """Update time series plot when controls change."""
    instance_id = plot_id["aio_id"]
    instance = TemplateTimeSeriesAIO.get_instance(instance_id)
    if instance is None or not isinstance(instance, TemplateTimeSeriesAIO):
        return go.Figure()

    return instance.build_figure(
        state_var=int(state_var),
        time_span=(time_range[0], time_range[1]),
        selected_templates=templates,
    )
