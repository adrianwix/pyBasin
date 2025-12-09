# pyright: basic
"""Phase plot page for template trajectories."""

from collections.abc import Sequence
from typing import cast

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
import torch
from dash import Input, Output, callback, dcc, html

from pybasin.cluster_classifier import SupervisedClassifier
from pybasin.plotters.base_page import BasePage
from pybasin.plotters.ids import IDs
from pybasin.plotters.types import PhasePlotOptions
from pybasin.plotters.utils import get_color

# Global page instances for callback access
_page_instance_2d: "PhasePlotPage | None" = None
_page_instance_3d: "PhasePlotPage | None" = None


def set_page_instance(page: "PhasePlotPage") -> None:
    """Set the global page instance for callback access.

    :param page: PhasePlotPage instance (2D or 3D).
    """
    global _page_instance_2d, _page_instance_3d  # noqa: PLW0603
    if page.is_3d:
        _page_instance_3d = page
    else:
        _page_instance_2d = page


class PhasePlotPage(BasePage):
    """2D or 3D phase plot of template trajectories."""

    # Component IDs for 2D version
    X_SELECT_2D = IDs.id(IDs.PHASE_2D, "x-select")
    Y_SELECT_2D = IDs.id(IDs.PHASE_2D, "y-select")
    TEMPLATE_SELECT_2D = IDs.id(IDs.PHASE_2D, "template-select")
    PLOT_2D = IDs.id(IDs.PHASE_2D, "plot")
    CONTROLS_2D = IDs.id(IDs.PHASE_2D, "controls")

    # Component IDs for 3D version
    X_SELECT_3D = IDs.id(IDs.PHASE_3D, "x-select")
    Y_SELECT_3D = IDs.id(IDs.PHASE_3D, "y-select")
    Z_SELECT_3D = IDs.id(IDs.PHASE_3D, "z-select")
    TEMPLATE_SELECT_3D = IDs.id(IDs.PHASE_3D, "template-select")
    PLOT_3D = IDs.id(IDs.PHASE_3D, "plot")
    CONTROLS_3D = IDs.id(IDs.PHASE_3D, "controls")

    def __init__(
        self,
        bse: object,
        state_labels: dict[int, str] | None = None,
        is_3d: bool = False,
        options: PhasePlotOptions | None = None,
        id_suffix: str | None = None,
    ):
        super().__init__(bse, state_labels)  # type: ignore[arg-type]
        self.is_3d = is_3d
        self.options = options or PhasePlotOptions()

    @property
    def page_id(self) -> str:
        return IDs.PHASE_3D if self.is_3d else IDs.PHASE_2D

    @property
    def nav_label(self) -> str:
        return "Phase Plot 3D" if self.is_3d else "Phase Plot 2D"

    @property
    def nav_icon(self) -> str:
        return "🌀" if self.is_3d else "〰️"

    def _get_template_labels(self) -> list[str]:
        """Get list of template labels if using SupervisedClassifier."""
        if isinstance(self.bse.cluster_classifier, SupervisedClassifier):
            return list(self.bse.cluster_classifier.labels)
        return []

    def build_layout(self) -> html.Div:
        """Build complete page layout with controls and plot.

        :return: Div containing controls and phase plot.
        """
        n_states = self.get_n_states()
        state_options = self.get_state_options()
        all_template_labels = self._get_template_labels()
        template_labels = list(all_template_labels)

        # Apply template filtering from options
        selected_templates = self.options.filter_templates(all_template_labels)

        if self.is_3d:
            controls_id = self.CONTROLS_3D
            plot_id = self.PLOT_3D
            x_id = self.X_SELECT_3D
            y_id = self.Y_SELECT_3D
            z_id = self.Z_SELECT_3D
            template_id = self.TEMPLATE_SELECT_3D
            z_var = self.options.get_z_var(n_states)
            default_z = str(z_var) if n_states > 2 else "0"
        else:
            controls_id = self.CONTROLS_2D
            plot_id = self.PLOT_2D
            x_id = self.X_SELECT_2D
            y_id = self.Y_SELECT_2D
            z_id = None
            template_id = self.TEMPLATE_SELECT_2D
            default_z = None

        # Use options for default x/y values
        default_x = str(self.options.x_var)
        default_y = str(self.options.y_var) if n_states > 1 else "0"

        controls: list[dmc.Select | dmc.MultiSelect] = []
        select_data = cast(Sequence[str], state_options)
        controls.append(
            dmc.Select(
                id=x_id,
                label="X Axis",
                data=select_data,
                value=default_x,
                w=150,
            )
        )
        controls.append(
            dmc.Select(
                id=y_id,
                label="Y Axis",
                data=select_data,
                value=default_y,
                w=150,
            )
        )

        if self.is_3d and z_id is not None and n_states > 2:
            controls.append(
                dmc.Select(
                    id=z_id,
                    label="Z Axis",
                    data=select_data,
                    value=default_z,
                    w=150,
                )
            )

        if template_labels:
            controls.append(
                dmc.MultiSelect(
                    id=template_id,
                    label="Templates",
                    data=template_labels,
                    value=selected_templates,
                    w=300,
                )
            )

        return html.Div(
            [
                dmc.Paper(
                    dmc.Group(controls, gap="md", id=controls_id),
                    p="md",
                    withBorder=True,
                    mb="md",
                ),
                dmc.Paper(
                    [
                        dcc.Graph(
                            id=plot_id,
                            figure=self.build_figure(
                                x_var=int(default_x),
                                y_var=int(default_y),
                                z_var=int(default_z) if default_z else None,
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
        x_var: int = 0,
        y_var: int = 1,
        z_var: int | None = None,
        selected_templates: list[str] | None = None,
        **kwargs: object,
    ) -> go.Figure:
        if not isinstance(self.bse.cluster_classifier, SupervisedClassifier):
            fig = go.Figure()
            fig.add_annotation(
                text="Phase plot requires SupervisedClassifier with template ICs",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        solver = self.bse.cluster_classifier.solver or self.bse.solver
        template_tensor = torch.tensor(
            self.bse.cluster_classifier.template_y0,
            dtype=torch.float32,
            device=solver.device,
        )

        _t, trajectories = solver.integrate(self.bse.ode_system, template_tensor)
        trajectories = trajectories.cpu().numpy()

        fig = go.Figure()

        all_labels = self.bse.cluster_classifier.labels
        if selected_templates is None:
            selected_templates = list(all_labels)

        for i, (label, traj) in enumerate(
            zip(
                all_labels,
                np.transpose(trajectories, (1, 0, 2)),
                strict=True,
            )
        ):
            if label not in selected_templates:
                continue
            if self.is_3d and z_var is not None:
                fig.add_trace(
                    go.Scatter3d(
                        x=traj[:, x_var],
                        y=traj[:, y_var],
                        z=traj[:, z_var],
                        mode="lines",
                        name=str(label),
                        line={"color": get_color(i), "width": 3},
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=traj[:, x_var],
                        y=traj[:, y_var],
                        mode="lines",
                        name=str(label),
                        line={"color": get_color(i), "width": 2},
                    )
                )

        if self.is_3d and z_var is not None:
            fig.update_layout(
                title="3D Phase Plot (Templates)",
                scene={
                    "xaxis_title": self.get_state_label(x_var),
                    "yaxis_title": self.get_state_label(y_var),
                    "zaxis_title": self.get_state_label(z_var),
                },
            )
        else:
            fig.update_layout(
                title="2D Phase Plot (Templates)",
                xaxis_title=self.get_state_label(x_var),
                yaxis_title=self.get_state_label(y_var),
            )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        return fig


# ============================================================================
# Module-level callbacks for 2D phase plot
# ============================================================================


@callback(
    Output(PhasePlotPage.PLOT_2D, "figure"),
    Input(PhasePlotPage.X_SELECT_2D, "value"),
    Input(PhasePlotPage.Y_SELECT_2D, "value"),
    Input(PhasePlotPage.TEMPLATE_SELECT_2D, "value"),
    prevent_initial_call=True,
)
def update_2d_figure(x_var: str, y_var: str, templates: list[str]) -> go.Figure:
    """Update 2D phase plot when controls change."""
    if _page_instance_2d is None:
        return go.Figure()

    return _page_instance_2d.build_figure(
        x_var=int(x_var),
        y_var=int(y_var),
        selected_templates=templates,
    )


# ============================================================================
# Module-level callbacks for 3D phase plot
# ============================================================================


@callback(
    Output(PhasePlotPage.PLOT_3D, "figure"),
    Input(PhasePlotPage.X_SELECT_3D, "value"),
    Input(PhasePlotPage.Y_SELECT_3D, "value"),
    Input(PhasePlotPage.Z_SELECT_3D, "value"),
    Input(PhasePlotPage.TEMPLATE_SELECT_3D, "value"),
    prevent_initial_call=True,
)
def update_3d_figure(x_var: str, y_var: str, z_var: str, templates: list[str]) -> go.Figure:
    """Update 3D phase plot when controls change."""
    if _page_instance_3d is None:
        return go.Figure()

    return _page_instance_3d.build_figure(
        x_var=int(x_var),
        y_var=int(y_var),
        z_var=int(z_var),
        selected_templates=templates,
    )
