"""AIO Parameter Bifurcation page showing amplitude evolution with k-means."""

from typing import Any

import dash_mantine_components as dmc  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
import plotly.graph_objects as go  # pyright: ignore[reportMissingTypeStubs]
import torch
from dash import (
    MATCH,
    Input,
    Output,
    State,
    callback,  # pyright: ignore[reportUnknownVariableType]
    dcc,
    html,
)
from plotly.subplots import (  # pyright: ignore[reportMissingTypeStubs]
    make_subplots,  # pyright: ignore[reportUnknownVariableType]
)
from sklearn.cluster import KMeans

from pybasin.as_basin_stability_estimator import ASBasinStabilityEstimator
from pybasin.plotters.interactive_plotter.ids_aio import aio_id
from pybasin.plotters.interactive_plotter.utils import get_color


class ParamBifurcationAIO:
    """
    AIO component for parameter bifurcation page.

    Shows amplitude evolution across parameter sweep with k-means clustering.
    """

    _instances: dict[str, "ParamBifurcationAIO"] = {}

    @classmethod
    def get_instance(cls, instance_id: str) -> "ParamBifurcationAIO | None":
        """Get instance by ID."""
        return cls._instances.get(instance_id)

    def __init__(
        self,
        as_bse: ASBasinStabilityEstimator,
        aio_id: str,
        state_labels: dict[int, str] | None = None,
    ):
        """
        Initialize parameter bifurcation AIO component.

        Args:
            as_bse: Adaptive study basin stability estimator
            aio_id: Unique identifier for this component instance
            state_labels: Optional mapping of state indices to labels
        """
        self.as_bse = as_bse
        self.aio_id = aio_id
        self.state_labels = state_labels or {}
        ParamBifurcationAIO._instances[aio_id] = self

    def get_state_label(self, idx: int) -> str:
        """Get label for a state variable."""
        return self.state_labels.get(idx, f"State {idx}")

    def get_n_states(self) -> int:
        """Get number of state variables."""
        if not self.as_bse.results:
            return 0
        first_result = self.as_bse.results[0]
        bifurc_amp = first_result.get("bifurcation_amplitudes")
        if bifurc_amp is None:
            return 0
        return bifurc_amp.shape[1]

    def get_state_options(self) -> list[dict[str, str]]:
        """Get dropdown options for state variable selection."""
        n_states = self.get_n_states()
        return [{"value": str(i), "label": self.get_state_label(i)} for i in range(n_states)]

    def _compute_amplitudes(
        self, bifurcation_amplitudes: torch.Tensor, dof: list[int], n_clusters: int
    ) -> np.ndarray:
        """Compute cluster centers for bifurcation amplitudes using k-means."""
        temp = bifurcation_amplitudes[:, dof]
        temp_np = temp.detach().cpu().numpy() if hasattr(temp, "detach") else np.asarray(temp)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(temp_np)

        return np.asarray(kmeans.cluster_centers_)

    def render(self) -> html.Div:
        """Render complete page layout with controls and plot."""
        state_options = self.get_state_options()

        return html.Div(
            [
                dmc.Paper(
                    [
                        dmc.MultiSelect(
                            id=aio_id("ParamBifurcation", self.aio_id, "dofs"),
                            label="State Dimensions",
                            data=state_options,  # type: ignore[arg-type]
                            value=["0"],
                            w=300,
                        ),
                    ],
                    p="md",
                    mb="md",
                    withBorder=True,
                ),
                dmc.Paper(
                    [
                        dcc.Graph(
                            id=aio_id("ParamBifurcation", self.aio_id, "plot"),
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
        """Build bifurcation diagram figure."""
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
                fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
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

        for j in range(n_dofs):
            fig.update_xaxes(  # pyright: ignore[reportUnknownMemberType]
                title="Parameter Value" if j == 0 else "",
                row=1,
                col=j + 1,
            )
            fig.update_yaxes(  # pyright: ignore[reportUnknownMemberType]
                title="Amplitude" if j == 0 else "",
                row=1,
                col=j + 1,
            )

        fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
            title="Bifurcation Diagram",
            template="plotly_dark",
            height=500,
        )

        return fig


@callback(
    Output(aio_id("ParamBifurcation", MATCH, "plot"), "figure"),
    Input(aio_id("ParamBifurcation", MATCH, "dofs"), "value"),
    State(aio_id("ParamBifurcation", MATCH, "plot"), "id"),
    prevent_initial_call=True,
)
def update_param_bifurcation_figure_aio(
    selected_dofs_str: list[str],
    plot_id: dict[str, Any],
) -> go.Figure:
    """Update bifurcation diagram when state selection changes."""
    instance_id = plot_id["aio_id"]
    instance = ParamBifurcationAIO.get_instance(instance_id)
    if instance is None:
        return go.Figure()

    selected_dofs = [int(dof) for dof in selected_dofs_str]
    return instance.build_figure(selected_dofs=selected_dofs)
