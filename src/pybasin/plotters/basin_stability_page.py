# pyright: basic
"""Basin stability bar chart page."""

import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import dcc, html

from pybasin.plotters.base_page import BasePage
from pybasin.plotters.ids import IDs
from pybasin.plotters.utils import get_color


class BasinStabilityPage(BasePage):
    """Bar chart showing basin stability values for each attractor."""

    # Component IDs using centralized registry
    PLOT = IDs.id(IDs.BASIN_STABILITY, "plot")
    INFO_PANEL = IDs.id(IDs.BASIN_STABILITY, "info-panel")

    def __init__(
        self,
        bse: object,
        state_labels: dict[int, str] | None = None,
        id_suffix: str = "",
    ):
        super().__init__(bse, state_labels)  # type: ignore[arg-type]
        self.id_suffix = id_suffix

        # Override class-level IDs with instance-specific ones
        if id_suffix:
            self.PLOT = f"{IDs.id(IDs.BASIN_STABILITY, 'plot')}-{id_suffix}"
            self.INFO_PANEL = f"{IDs.id(IDs.BASIN_STABILITY, 'info-panel')}-{id_suffix}"

    @property
    def page_id(self) -> str:
        return IDs.BASIN_STABILITY

    @property
    def nav_label(self) -> str:
        return "Basin Stability"

    @property
    def nav_icon(self) -> str:
        return "📊"

    def build_layout(self) -> html.Div:
        """Build complete page layout with info panel and bar chart.

        :return: Div containing info panel and bar chart.
        """
        return html.Div(
            [
                dmc.Grid(
                    [
                        # Info panel on the left
                        dmc.GridCol(
                            self.build_info_panel(),
                            span=4,
                            id=self.INFO_PANEL,
                        ),
                        # Bar chart on the right
                        dmc.GridCol(
                            dmc.Paper(
                                [
                                    dcc.Graph(
                                        id=self.PLOT,
                                        figure=self.build_figure(),
                                        style={"height": "60vh"},
                                        config={
                                            "displayModeBar": True,
                                            "scrollZoom": True,
                                        },
                                    ),
                                ],
                                p="md",
                                withBorder=True,
                            ),
                            span="auto",
                        ),
                    ],
                    gutter="md",
                ),
            ]
        )

    def build_controls(self) -> dmc.Group | None:
        return None

    def build_info_panel(self) -> dmc.Paper:
        """Build the information panel showing ODE system and sampler details."""
        ode_str = self.bse.ode_system.get_str()

        sampler = self.bse.sampler
        min_limits = sampler.min_limits.cpu().tolist()
        max_limits = sampler.max_limits.cpu().tolist()

        state_ranges = []
        for i, (min_val, max_val) in enumerate(zip(min_limits, max_limits, strict=True)):
            label = self.get_state_label(i)
            state_ranges.append(
                dmc.Text(
                    f"{label}: [{min_val:.4g}, {max_val:.4g}]",
                    size="sm",
                    ff="monospace",
                )
            )

        ode_params_section = []
        if hasattr(self.bse.ode_system, "params"):
            params = self.bse.ode_system.params
            if isinstance(params, dict):
                param_items = []
                for key, value in params.items():
                    formatted_value = f"{value:.4g}" if isinstance(value, float) else str(value)
                    param_items.append(
                        dmc.Group(
                            [
                                dmc.Text(f"{key}:", size="sm", fw="normal", ff="monospace"),
                                dmc.Text(formatted_value, size="sm", ff="monospace"),
                            ],
                            gap="xs",
                        )
                    )
                ode_params_section = [
                    dmc.Text("Parameters", fw="bold", size="lg", mt="sm"),
                    dmc.Stack(param_items, gap="xs", ml="md"),
                ]

        return dmc.Paper(
            [
                dmc.Stack(
                    [
                        dmc.Text("ODE System", fw="bold", size="lg"),
                        dmc.Code(
                            ode_str,
                            block=True,
                        ),
                        *ode_params_section,
                        dmc.Divider(my="sm"),
                        dmc.Text("Sampler Configuration", fw="bold", size="lg"),
                        dmc.Group(
                            [
                                dmc.Text("Type:", size="sm", fw="normal"),
                                dmc.Badge(
                                    self.get_sampler_name(),
                                    color="blue",
                                    variant="light",
                                    style={"textTransform": "none"},
                                ),
                            ],
                            gap="xs",
                        ),
                        dmc.Text("State Ranges:", size="sm", fw="normal", mt="xs"),
                        dmc.Stack(state_ranges, gap="xs", ml="md"),
                        dmc.Divider(my="sm"),
                        dmc.Text("Analysis Configuration", fw="bold", size="lg"),
                        dmc.Group(
                            [
                                dmc.Text("Samples:", size="sm", fw="normal"),
                                dmc.Badge(
                                    str(self.bse.n),
                                    color="green",
                                    variant="light",
                                    style={"textTransform": "none"},
                                ),
                            ],
                            gap="xs",
                        ),
                        dmc.Group(
                            [
                                dmc.Text("Solver:", size="sm", fw="normal"),
                                dmc.Badge(
                                    self.get_solver_name(),
                                    color="grape",
                                    variant="light",
                                    style={"textTransform": "none"},
                                ),
                            ],
                            gap="xs",
                        ),
                        dmc.Group(
                            [
                                dmc.Text("Classifier:", size="sm", fw="normal"),
                                dmc.Badge(
                                    self.get_classifier_name(),
                                    color="orange",
                                    variant="light",
                                    style={"textTransform": "none"},
                                ),
                            ],
                            gap="xs",
                        ),
                    ],
                    gap="xs",
                ),
            ],
            p="md",
            withBorder=True,
            style={"height": "100%"},
        )

    def build_figure(self, **kwargs: object) -> go.Figure:
        if self.bse.bs_vals is None:
            return go.Figure()

        labels = list(self.bse.bs_vals.keys())
        values = list(self.bse.bs_vals.values())
        colors = [get_color(i) for i in range(len(labels))]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=colors,
                    text=[f"{v:.2%}" for v in values],
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title="Basin Stability",
            xaxis_title="Attractor",
            yaxis_title="Fraction of Samples",
            yaxis_range=[0, 1],
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        return fig
