# pyright: basic
"""Template time series page."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
import torch
from dash import Input, Output, callback, dcc, html

from pybasin.cluster_classifier import SupervisedClassifier
from pybasin.plotters.base_page import BasePage
from pybasin.plotters.ids import IDs
from pybasin.plotters.types import TemplateTimeSeriesOptions
from pybasin.plotters.utils import get_color

if TYPE_CHECKING:
    from pybasin.basin_stability_estimator import BasinStabilityEstimator


# Global page instance for callback access
_page_instance: "TemplateTimeSeriesPage | None" = None


def set_page_instance(page: "TemplateTimeSeriesPage") -> None:
    """Set the global page instance for callback access.

    :param page: TemplateTimeSeriesPage instance.
    """
    global _page_instance  # noqa: PLW0603
    _page_instance = page


class TemplateTimeSeriesPage(BasePage):
    """Time series plot for template trajectories."""

    # Component IDs using centralized registry
    STATE_SELECT = IDs.id(IDs.TEMPLATE_TS, "state-select")
    TEMPLATE_SELECT = IDs.id(IDs.TEMPLATE_TS, "template-select")
    TIME_SLIDER = IDs.id(IDs.TEMPLATE_TS, "time-slider")
    PLOT = IDs.id(IDs.TEMPLATE_TS, "plot")
    CONTROLS = IDs.id(IDs.TEMPLATE_TS, "controls")

    def __init__(
        self,
        bse: "BasinStabilityEstimator",
        state_labels: dict[int, str] | None = None,
        options: TemplateTimeSeriesOptions | None = None,
    ):
        super().__init__(bse, state_labels)
        self.options = options or TemplateTimeSeriesOptions()
        self._cached_time: np.ndarray | None = None
        self._cached_trajectories: np.ndarray | None = None

    @property
    def page_id(self) -> str:
        return IDs.TEMPLATE_TS

    @property
    def nav_label(self) -> str:
        return "Time Series"

    @property
    def nav_icon(self) -> str:
        return "📉"

    def _ensure_integrated(self) -> tuple[np.ndarray, np.ndarray]:
        """Ensure template trajectories are integrated and cached."""
        if self._cached_time is not None and self._cached_trajectories is not None:
            return self._cached_time, self._cached_trajectories

        if not isinstance(self.bse.cluster_classifier, SupervisedClassifier):
            return np.array([0.0, 1.0]), np.zeros((2, 1, 1))

        solver = self.bse.cluster_classifier.solver or self.bse.solver
        template_tensor = torch.tensor(
            self.bse.cluster_classifier.template_y0,
            dtype=torch.float32,
            device=solver.device,
        )

        t, y = solver.integrate(self.bse.ode_system, template_tensor)
        self._cached_time = t.cpu().numpy()
        self._cached_trajectories = y.cpu().numpy()

        return self._cached_time, self._cached_trajectories

    def get_time_bounds(self) -> tuple[float, float]:
        """Get the time bounds from the integrated solution."""
        t, _ = self._ensure_integrated()
        return float(t[0]), float(t[-1])

    def get_default_time_span(self) -> tuple[float, float]:
        """Get default time span based on options."""
        t_min, t_max = self.get_time_bounds()
        return t_min, t_min + (t_max - t_min) * self.options.time_range_percent

    def get_template_options(self) -> list[str]:
        """Get template labels for the multi-select component."""
        return self.get_all_template_labels()

    def get_all_template_labels(self) -> list[str]:
        """Get all template labels for default selection."""
        if not isinstance(self.bse.cluster_classifier, SupervisedClassifier):
            return []
        return list(self.bse.cluster_classifier.labels)

    def build_layout(self) -> html.Div:
        """Build complete page layout with controls and plot.

        :return: Div containing controls and time series plot.
        """
        state_options = self.get_state_options()
        template_options = self.get_template_options()
        all_templates = self.get_all_template_labels()
        t_min, t_max = self.get_time_bounds()
        default_span = self.get_default_time_span()

        # Apply template filtering from options
        selected_templates = self.options.filter_templates(all_templates)
        state_var = self.options.state_var

        select_data = cast(Sequence[str], state_options)
        controls: list[dmc.Select | dmc.MultiSelect] = []
        controls.append(
            dmc.Select(
                id=self.STATE_SELECT,
                label="State Variable",
                data=select_data,
                value=str(state_var),
                w=150,
            )
        )

        if template_options:
            controls.append(
                dmc.MultiSelect(
                    id=self.TEMPLATE_SELECT,
                    label="Templates",
                    data=template_options,
                    value=selected_templates,
                    w=300,
                )
            )

        return html.Div(
            [
                dmc.Paper(
                    [
                        dmc.Group(controls, gap="md", id=self.CONTROLS),
                        dmc.Space(h="md"),
                        dmc.Text("Time Range", size="sm", fw="normal"),
                        dcc.RangeSlider(
                            id=self.TIME_SLIDER,
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
                            id=self.PLOT,
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
            ]
        )

    def build_controls(self) -> dmc.Group | None:
        return None

    def build_figure(
        self,
        state_var: int = 0,
        time_span: tuple[float, float] | None = None,
        selected_templates: list[str] | None = None,
        **kwargs: object,
    ) -> go.Figure:
        if not isinstance(self.bse.cluster_classifier, SupervisedClassifier):
            fig = go.Figure()
            fig.add_annotation(
                text="Template plot requires SupervisedClassifier",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        t, y = self._ensure_integrated()

        if time_span is not None:
            mask = (t >= time_span[0]) & (t <= time_span[1])
            t = t[mask]
            y = y[mask]

        fig = go.Figure()

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
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=traj[:, state_var],
                    mode="lines",
                    name=str(label),
                    line={"color": get_color(i), "width": 2},
                )
            )

        fig.update_layout(
            title=f"Template Trajectories - {self.get_state_label(state_var)}",
            xaxis_title="Time",
            yaxis_title=self.get_state_label(state_var),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        return fig


# ============================================================================
# Module-level callbacks
# ============================================================================


@callback(
    Output(TemplateTimeSeriesPage.PLOT, "figure"),
    Input(TemplateTimeSeriesPage.STATE_SELECT, "value"),
    Input(TemplateTimeSeriesPage.TEMPLATE_SELECT, "value"),
    Input(TemplateTimeSeriesPage.TIME_SLIDER, "value"),
    prevent_initial_call=True,
)
def update_figure(state_var: str, templates: list[str], time_range: list[float]) -> go.Figure:
    """Update time series plot when controls change."""
    if _page_instance is None:
        return go.Figure()

    return _page_instance.build_figure(
        state_var=int(state_var),
        time_span=(time_range[0], time_range[1]) if time_range else None,
        selected_templates=templates,
    )
