# pyright: basic
"""Parameter Bifurcation page showing amplitude evolution with k-means clustering."""

import dash_mantine_components as dmc  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import plotly.graph_objects as go
import torch
from dash import Input, Output, callback, dcc, html
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans

from pybasin.as_basin_stability_estimator import ASBasinStabilityEstimator
from pybasin.plotters.as_base_page import ASBasePage
from pybasin.plotters.ids import IDs
from pybasin.plotters.utils import get_color

_page_instance: "ParamBifurcationPage | None" = None


def set_page_instance(page: "ParamBifurcationPage") -> None:
    """Set the global page instance for callbacks."""
    global _page_instance
    _page_instance = page


class ParamBifurcationPage(ASBasePage):
    """Parameter Bifurcation page showing amplitude evolution with k-means clustering."""

    PLOT = IDs.id(IDs.PARAM_BIFURCATION, "plot")
    DOFS = IDs.id(IDs.PARAM_BIFURCATION, "dofs")
    CONTROLS = IDs.id(IDs.PARAM_BIFURCATION, "controls")

    def __init__(
        self,
        as_bse: ASBasinStabilityEstimator,
        state_labels: dict[int, str] | None = None,
    ):
        super().__init__(as_bse, state_labels)

    @property
    def page_id(self) -> str:
        return IDs.PARAM_BIFURCATION

    @property
    def nav_label(self) -> str:
        return "Bifurcation"

    @property
    def nav_icon(self) -> str:
        return "🌀"

    def _compute_amplitudes(
        self, bifurcation_amplitudes: torch.Tensor, dof: list[int], n_clusters: int
    ) -> np.ndarray:
        """Compute cluster centers for bifurcation amplitudes using k-means.

        :param bifurcation_amplitudes: Tensor of shape (n_samples, n_states).
        :param dof: List of degrees of freedom (state indices) to analyze.
        :param n_clusters: Number of clusters for k-means.
        :return: Array of cluster centers, shape (n_clusters, n_dofs).
        """
        temp = bifurcation_amplitudes[:, dof]
        temp_np = temp.detach().cpu().numpy() if hasattr(temp, "detach") else np.asarray(temp)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(temp_np)

        return np.asarray(kmeans.cluster_centers_)

    def build_layout(self) -> html.Div:
        """Build the parameter bifurcation page layout."""
        state_options = self.get_state_options()

        return html.Div(
            [
                dmc.Paper(
                    [
                        dmc.MultiSelect(
                            id=self.DOFS,
                            label="State Dimensions",
                            data=state_options,  # pyright: ignore[reportArgumentType]
                            value=["0"],
                            w=300,
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
                            figure=self.build_figure([0]),
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

    def build_figure(self, selected_dofs: list[int]) -> go.Figure:
        """Build the bifurcation diagram figure.

        :param selected_dofs: List of state indices to plot.
        :return: Plotly figure object.
        """
        n_clusters = self.get_n_states()
        n_dofs = len(selected_dofs)
        n_par_var = len(self.as_bse.results)

        amplitudes = np.zeros((n_clusters, n_dofs, n_par_var))

        for idx, result in enumerate(self.as_bse.results):
            bifurcation_amplitudes = result["bifurcation_amplitudes"]
            if bifurcation_amplitudes is None:
                raise ValueError(
                    f"Missing bifurcation amplitudes for parameter {result['param_value']}"
                )

            centers = self._compute_amplitudes(bifurcation_amplitudes, selected_dofs, n_clusters)
            amplitudes[:, :, idx] = centers

        fig = make_subplots(
            rows=1,
            cols=n_dofs,
            subplot_titles=[self.get_state_label(d) for d in selected_dofs],
            shared_yaxes=True,
        )

        for j in range(n_dofs):
            for i in range(n_clusters):
                fig.add_trace(
                    go.Scatter(
                        x=self.as_bse.parameter_values,
                        y=amplitudes[i, j, :],
                        mode="lines+markers",
                        name=f"Cluster {i + 1}",
                        line={"color": get_color(i)},
                        marker={"size": 8},
                        showlegend=(j == 0),
                    ),
                    row=1,
                    col=j + 1,
                )

        fig.update_xaxes(title_text=self.get_parameter_name_short())
        fig.update_yaxes(title_text="Amplitude", col=1)

        fig.update_layout(
            title="Bifurcation Diagram",
            hovermode="x unified",
            template="plotly_dark",
            height=600,
        )

        return fig


@callback(
    Output(ParamBifurcationPage.PLOT, "figure"),
    Input(ParamBifurcationPage.DOFS, "value"),
    prevent_initial_call=True,
)
def update_param_bifurcation_figure(selected_dofs_str: list[str]) -> go.Figure:
    """Update the parameter bifurcation figure based on selected DOFs."""
    if _page_instance is None:
        return go.Figure()

    selected_dofs = [int(d) for d in selected_dofs_str]
    if not selected_dofs:
        selected_dofs = [0]

    return _page_instance.build_figure(selected_dofs=selected_dofs)
