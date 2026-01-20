"""AIO Parameter Overview page showing basin stability across parameter values."""

from typing import Any

import dash_mantine_components as dmc  # pyright: ignore[reportMissingTypeStubs]
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

from pybasin.as_basin_stability_estimator import ASBasinStabilityEstimator
from pybasin.plotters.interactive_plotter.ids_aio import aio_id
from pybasin.plotters.interactive_plotter.utils import get_color


class ParamOverviewAIO:
    """
    AIO component for parameter overview page.

    Shows basin stability evolution across parameter sweep values.
    """

    _instances: dict[str, "ParamOverviewAIO"] = {}

    @classmethod
    def get_instance(cls, instance_id: str) -> "ParamOverviewAIO | None":
        """Get instance by ID."""
        return cls._instances.get(instance_id)

    def __init__(
        self,
        as_bse: ASBasinStabilityEstimator,
        aio_id: str,
        state_labels: dict[int, str] | None = None,
    ):
        """
        Initialize parameter overview AIO component.

        :param as_bse: Adaptive study basin stability estimator.
        :param aio_id: Unique identifier for this component instance.
        :param state_labels: Optional mapping of state indices to labels.
        """
        self.as_bse = as_bse
        self.aio_id = aio_id
        self.state_labels = state_labels or {}
        ParamOverviewAIO._instances[aio_id] = self

    def get_parameter_name_short(self) -> str:
        """Get short parameter name."""
        return self.as_bse.as_params["adaptative_parameter_name"].split(".")[-1]

    def _get_all_labels(self) -> list[str]:
        """Get all unique labels across all parameter values."""
        all_labels: set[str] = set()
        for bs_dict in self.as_bse.basin_stabilities:
            all_labels.update(bs_dict.keys())
        return sorted(all_labels)

    def render(self) -> html.Div:
        """Render complete page layout with controls and plot."""
        all_labels = self._get_all_labels()
        label_options = [{"value": label, "label": label} for label in all_labels]

        return html.Div(
            [
                dmc.Paper(
                    [
                        dmc.Group(
                            [
                                dmc.SegmentedControl(
                                    id=aio_id("ParamOverview", self.aio_id, "scale"),
                                    data=[  # pyright: ignore[reportArgumentType]
                                        {"value": "linear", "label": "Linear"},
                                        {"value": "log", "label": "Log"},
                                    ],
                                    value="linear",
                                ),
                                dmc.MultiSelect(
                                    id=aio_id("ParamOverview", self.aio_id, "labels"),
                                    label="Show Labels",
                                    # Typing is bad: https://www.dash-mantine-components.com/components/multiselect
                                    data=label_options,  # pyright: ignore[reportArgumentType]
                                    value=all_labels,
                                    w=300,
                                ),
                            ],
                            gap="md",
                        ),
                    ],
                    p="md",
                    mb="md",
                    withBorder=True,
                ),
                dmc.Paper(
                    [
                        dcc.Graph(
                            id=aio_id("ParamOverview", self.aio_id, "plot"),
                            figure=self.build_figure(),
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
            ]
        )

    def build_figure(
        self, x_scale: str = "linear", label_filter: list[str] | None = None
    ) -> go.Figure:
        """Build basin stability variation figure."""
        all_labels = self._get_all_labels()
        labels_to_show = label_filter if label_filter else all_labels

        bs_values: dict[str, list[float]] = {label: [] for label in all_labels}
        for bs_dict in self.as_bse.basin_stabilities:
            for label in all_labels:
                bs_values[label].append(bs_dict.get(label, 0))

        fig = go.Figure()

        for i, label in enumerate(all_labels):
            if label in labels_to_show:
                fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
                    go.Scatter(
                        x=self.as_bse.parameter_values,
                        y=bs_values[label],
                        mode="lines+markers",
                        name=label,
                        line={"color": get_color(i)},
                        marker={"size": 8},
                    )
                )

        fig.update_xaxes(  # pyright: ignore[reportUnknownMemberType]
            type="log" if x_scale == "log" else "linear",
            title=self.get_parameter_name_short(),
        )
        fig.update_yaxes(title="Basin Stability", range=[0, 1])  # pyright: ignore[reportUnknownMemberType]
        fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
            title="Basin Stability vs Parameter Variation",
            hovermode="x unified",
            template="plotly_dark",
            legend={
                "orientation": "v",
                "yanchor": "top",
                "y": 1,
                "xanchor": "left",
                "x": 1.02,
            },
        )

        return fig


@callback(
    Output(aio_id("ParamOverview", MATCH, "plot"), "figure"),
    [
        Input(aio_id("ParamOverview", MATCH, "scale"), "value"),
        Input(aio_id("ParamOverview", MATCH, "labels"), "value"),
    ],
    State(aio_id("ParamOverview", MATCH, "plot"), "id"),
    prevent_initial_call=True,
)
def update_param_overview_figure_aio(
    x_scale: str,
    label_filter: list[str],
    plot_id: dict[str, Any],
) -> go.Figure:
    """Update parameter overview figure when controls change."""
    instance_id = plot_id["aio_id"]
    instance = ParamOverviewAIO.get_instance(instance_id)
    if instance is None:
        return go.Figure()

    return instance.build_figure(x_scale=x_scale, label_filter=label_filter)
