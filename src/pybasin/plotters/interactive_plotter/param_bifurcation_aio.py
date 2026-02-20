"""AIO Parameter Bifurcation page showing amplitude evolution with k-means."""

from collections import defaultdict
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

from pybasin.basin_stability_study import BasinStabilityStudy
from pybasin.plotters.interactive_plotter.ids_aio import aio_id
from pybasin.plotters.interactive_plotter.utils import get_color


class ParamBifurcationAIO:
    """
    AIO component for parameter bifurcation page.

    Shows amplitude evolution across parameter sweep with k-means clustering.
    Supports multi-parameter studies by grouping results along one chosen parameter
    (x-axis) while producing separate curves for each combination of the remaining
    parameters.
    """

    _instances: dict[str, "ParamBifurcationAIO"] = {}

    @classmethod
    def get_instance(cls, instance_id: str) -> "ParamBifurcationAIO | None":
        """Get instance by ID."""
        return cls._instances.get(instance_id)

    def __init__(
        self,
        bs_study: BasinStabilityStudy,
        aio_id: str,
        state_labels: dict[int, str] | None = None,
    ):
        """
        Initialize parameter bifurcation AIO component.

        :param bs_study: Basin stability study instance.
        :param aio_id: Unique identifier for this component instance.
        :param state_labels: Optional mapping of state indices to labels.
        """
        self.bs_study = bs_study
        self.aio_id = aio_id
        self.state_labels = state_labels or {}
        ParamBifurcationAIO._instances[aio_id] = self

    def get_state_label(self, idx: int) -> str:
        """Get label for a state variable."""
        return self.state_labels.get(idx, f"State {idx}")

    def get_n_states(self) -> int:
        """Get number of state variables."""
        if not self.bs_study.results:
            return 0
        first_result = self.bs_study.results[0]
        bifurc_amp = first_result.get("bifurcation_amplitudes")
        if bifurc_amp is None:
            return 0
        return bifurc_amp.shape[1]

    def get_state_options(self) -> list[dict[str, str]]:
        """Get dropdown options for state variable selection."""
        n_states = self.get_n_states()
        return [{"value": str(i), "label": self.get_state_label(i)} for i in range(n_states)]

    def get_parameter_names(self) -> list[str]:
        """Get all studied parameter names."""
        return self.bs_study.studied_parameter_names

    def get_parameter_options(self) -> list[dict[str, str]]:
        """Get dropdown options for parameter selection."""
        return [{"value": name, "label": name} for name in self.get_parameter_names()]

    def _group_by_parameter(self, param_name: str) -> dict[tuple[tuple[str, Any], ...], list[int]]:
        """Group study result indices by the values of all parameters except param_name.

        :param param_name: The parameter whose values form the x-axis.
        :return: Mapping from a tuple of (other_param, value) pairs to sorted result indices.
        """
        other_params = [p for p in self.get_parameter_names() if p != param_name]

        groups: dict[tuple[tuple[str, Any], ...], list[int]] = defaultdict(list)
        for i, sl in enumerate(self.bs_study.study_labels):
            group_key = tuple((p, sl[p]) for p in other_params) if other_params else ()
            groups[group_key].append(i)

        for group_key in groups:
            groups[group_key].sort(key=lambda i: self.bs_study.study_labels[i][param_name])

        return dict(groups)

    def _compute_amplitudes(
        self, bifurcation_amplitudes: torch.Tensor, dof: list[int], n_clusters: int
    ) -> np.ndarray:
        """Compute cluster centers for bifurcation amplitudes using k-means."""
        temp = bifurcation_amplitudes[:, dof]
        temp_np = temp.detach().cpu().numpy() if hasattr(temp, "detach") else np.asarray(temp)

        # Handle cases with fewer samples than clusters
        n_samples = len(temp_np)
        actual_n_clusters = min(n_clusters, n_samples)

        if actual_n_clusters == 0:
            return np.zeros((n_clusters, len(dof)))

        kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42)
        kmeans.fit(temp_np)

        centers = np.asarray(kmeans.cluster_centers_)

        # Pad if fewer clusters than expected
        if actual_n_clusters < n_clusters:
            centers_padded = np.zeros((n_clusters, len(dof)))
            centers_padded[:actual_n_clusters] = centers
            return centers_padded

        return centers

    def render(self) -> html.Div:
        """Render complete page layout with controls and plot."""
        state_options = self.get_state_options()
        param_options = self.get_parameter_options()
        default_param = param_options[0]["value"] if param_options else ""

        return html.Div(
            [
                dmc.Paper(
                    [
                        dmc.Group(
                            [
                                dmc.Select(
                                    id=aio_id("ParamBifurcation", self.aio_id, "x_param"),
                                    label="X-Axis Parameter",
                                    data=param_options,  # type: ignore[arg-type]
                                    value=default_param,
                                    w=200,
                                ),
                                dmc.MultiSelect(
                                    id=aio_id("ParamBifurcation", self.aio_id, "dofs"),
                                    label="State Dimensions",
                                    data=state_options,  # type: ignore[arg-type]
                                    value=["0"],
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
                            id=aio_id("ParamBifurcation", self.aio_id, "plot"),
                            figure=self.build_figure(x_param=default_param, selected_dofs=[0]),
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
        self,
        x_param: str | None = None,
        selected_dofs: list[int] | None = None,
    ) -> go.Figure:
        """Build bifurcation diagram figure.

        :param x_param: Parameter to use as x-axis. Defaults to first parameter.
        :param selected_dofs: List of state indices to plot. Defaults to [0].
        :return: Plotly figure.
        """
        params = self.get_parameter_names()
        if not params:
            return go.Figure()

        if x_param is None or x_param not in params:
            x_param = params[0]

        if selected_dofs is None:
            selected_dofs = [0]

        n_clusters = self.get_n_states()
        n_dofs = len(selected_dofs)

        groups = self._group_by_parameter(x_param)
        n_groups = len(groups)

        fig = make_subplots(
            rows=1,
            cols=n_dofs,
            subplot_titles=[self.get_state_label(d) for d in selected_dofs],
            shared_yaxes=True,
        )

        # Colors: use group color if multiple groups, else cluster color
        for g_idx, (group_key, indices) in enumerate(groups.items()):
            x_values = [self.bs_study.study_labels[i][x_param] for i in indices]
            n_par_var = len(indices)

            amplitudes = np.zeros((n_clusters, n_dofs, n_par_var))

            for pos, result_idx in enumerate(indices):
                result = self.bs_study.results[result_idx]
                bifurcation_amplitudes = result["bifurcation_amplitudes"]
                if bifurcation_amplitudes is None:
                    continue

                centers = self._compute_amplitudes(
                    bifurcation_amplitudes, selected_dofs, n_clusters
                )
                amplitudes[:, :, pos] = centers

            group_suffix = ""
            if group_key:
                group_suffix = " (" + ", ".join(f"{k}={v}" for k, v in group_key) + ")"

            for j in range(n_dofs):
                for i in range(n_clusters):
                    # Use group index for color if multiple groups, else cluster index
                    color_idx = g_idx if n_groups > 1 else i
                    fig.add_trace(  # pyright: ignore[reportUnknownMemberType]
                        go.Scatter(
                            x=x_values,
                            y=amplitudes[i, j, :],
                            mode="lines+markers",
                            name=f"Cluster {i + 1}{group_suffix}",
                            line={"color": get_color(color_idx)},
                            marker={"size": 8},
                            showlegend=(j == 0),
                            legendgroup=f"group_{g_idx}_cluster_{i}",
                        ),
                        row=1,
                        col=j + 1,
                    )

        for j in range(n_dofs):
            fig.update_xaxes(  # pyright: ignore[reportUnknownMemberType]
                title=x_param if j == 0 else "",
                autorange=True,
                row=1,
                col=j + 1,
            )
            fig.update_yaxes(  # pyright: ignore[reportUnknownMemberType]
                title="Amplitude" if j == 0 else "",
                autorange=True,
                row=1,
                col=j + 1,
            )

        fig.update_layout(  # pyright: ignore[reportUnknownMemberType]
            title=f"Bifurcation Diagram ({x_param})",
            template="plotly_dark",
            height=500,
        )

        return fig


@callback(
    Output(aio_id("ParamBifurcation", MATCH, "plot"), "figure"),
    [
        Input(aio_id("ParamBifurcation", MATCH, "x_param"), "value"),
        Input(aio_id("ParamBifurcation", MATCH, "dofs"), "value"),
    ],
    State(aio_id("ParamBifurcation", MATCH, "plot"), "id"),
    prevent_initial_call=True,
)
def update_param_bifurcation_figure_aio(
    x_param: str,
    selected_dofs_str: list[str],
    plot_id: dict[str, Any],
) -> go.Figure:
    """Update bifurcation diagram when controls change."""
    instance_id = plot_id["aio_id"]
    instance = ParamBifurcationAIO.get_instance(instance_id)
    if instance is None:
        return go.Figure()

    selected_dofs = [int(dof) for dof in selected_dofs_str]
    return instance.build_figure(x_param=x_param, selected_dofs=selected_dofs)
