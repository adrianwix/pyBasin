# pyright: basic
"""Parameter Overview page showing basin stability variation across parameters."""

import dash_mantine_components as dmc  # pyright: ignore[reportMissingTypeStubs]
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from pybasin.as_basin_stability_estimator import ASBasinStabilityEstimator
from pybasin.plotters.as_base_page import ASBasePage
from pybasin.plotters.ids import IDs
from pybasin.plotters.utils import get_color

_page_instance: "ParamOverviewPage | None" = None


def set_page_instance(page: "ParamOverviewPage") -> None:
    """Set the global page instance for callbacks."""
    global _page_instance
    _page_instance = page


class ParamOverviewPage(ASBasePage):
    """Parameter Overview page showing basin stability evolution across parameter values."""

    PLOT = IDs.id(IDs.PARAM_OVERVIEW, "plot")
    SCALE = IDs.id(IDs.PARAM_OVERVIEW, "scale")
    LABELS = IDs.id(IDs.PARAM_OVERVIEW, "labels")
    CONTROLS = IDs.id(IDs.PARAM_OVERVIEW, "controls")

    def __init__(
        self,
        as_bse: ASBasinStabilityEstimator,
        state_labels: dict[int, str] | None = None,
    ):
        super().__init__(as_bse, state_labels)

    @property
    def page_id(self) -> str:
        return IDs.PARAM_OVERVIEW

    @property
    def nav_label(self) -> str:
        return "Overview"

    @property
    def nav_icon(self) -> str:
        return "📊"

    def _get_all_labels(self) -> list[str]:
        """Get all unique labels across all parameter values."""
        all_labels = set()
        for bs_dict in self.as_bse.basin_stabilities:
            all_labels.update(bs_dict.keys())
        return sorted(all_labels)

    def build_layout(self) -> html.Div:
        """Build the parameter overview page layout."""
        all_labels = self._get_all_labels()
        label_options = [{"value": label, "label": label} for label in all_labels]

        return html.Div(
            [
                dmc.Paper(
                    [
                        dmc.Group(
                            [
                                dmc.SegmentedControl(
                                    id=self.SCALE,
                                    data=[
                                        {"value": "linear", "label": "Linear"},
                                        {"value": "log", "label": "Log"},
                                    ],
                                    value="linear",
                                ),
                                dmc.MultiSelect(
                                    id=self.LABELS,
                                    label="Show Labels",
                                    data=label_options,
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
                    id=self.CONTROLS,
                ),
                dmc.Paper(
                    [
                        dcc.Graph(
                            id=self.PLOT,
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
        """Build the basin stability variation figure.

        :param x_scale: X-axis scale type ("linear" or "log").
        :param label_filter: List of labels to show (None = show all).
        :return: Plotly figure object.
        """
        all_labels = self._get_all_labels()
        labels_to_show = label_filter if label_filter else all_labels

        bs_values: dict[str, list[float]] = {label: [] for label in all_labels}
        for bs_dict in self.as_bse.basin_stabilities:
            for label in all_labels:
                bs_values[label].append(bs_dict.get(label, 0))

        fig = go.Figure()

        for i, label in enumerate(all_labels):
            if label in labels_to_show:
                fig.add_trace(
                    go.Scatter(
                        x=self.as_bse.parameter_values,
                        y=bs_values[label],
                        mode="lines+markers",
                        name=label,
                        line={"color": get_color(i)},
                        marker={"size": 8},
                    )
                )

        fig.update_xaxes(
            type="log" if x_scale == "log" else "linear",
            title=self.get_parameter_name_short(),
        )
        fig.update_yaxes(title="Basin Stability", range=[0, 1])
        fig.update_layout(
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
    Output(ParamOverviewPage.PLOT, "figure"),
    [
        Input(ParamOverviewPage.SCALE, "value"),
        Input(ParamOverviewPage.LABELS, "value"),
    ],
    prevent_initial_call=True,
)
def update_param_overview_figure(x_scale: str, label_filter: list[str]) -> go.Figure:
    """Update the parameter overview figure based on control inputs."""
    if _page_instance is None:
        return go.Figure()
    return _page_instance.build_figure(x_scale=x_scale, label_filter=label_filter)
