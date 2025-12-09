# pyright: basic
"""Feature space scatter plot page with dynamic axis selection."""

from collections.abc import Sequence
from typing import Any, cast

import dash_mantine_components as dmc
import numpy as np
import plotly.graph_objects as go
from dash import Input, NoUpdate, Output, State, callback, dcc, html, no_update

from pybasin.plotters.base_page import BasePage
from pybasin.plotters.ids import IDs
from pybasin.plotters.trajectory_modal import SELECTED_SAMPLE_DATA, TrajectoryModal
from pybasin.plotters.types import FeatureSpaceOptions
from pybasin.plotters.utils import get_color, use_webgl


class FeatureSpacePage(BasePage):
    """Scatter plot of features with selectable axes.

    Supports dynamic feature axis selection based on the feature extractor's
    configuration. For JaxFeatureExtractor, this includes per-state features
    with human-readable labels combining state names and feature names.
    """

    # Component IDs using centralized registry
    X_SELECT = IDs.id(IDs.FEATURE_SPACE, "x-select")
    Y_SELECT = IDs.id(IDs.FEATURE_SPACE, "y-select")
    Y_CONTAINER = IDs.id(IDs.FEATURE_SPACE, "y-container")
    LABEL_SELECT = IDs.id(IDs.FEATURE_SPACE, "label-select")
    FEATURE_TYPE_SWITCH = IDs.id(IDs.FEATURE_SPACE, "feature-type-switch")
    PLOT = IDs.id(IDs.FEATURE_SPACE, "plot")
    CONTROLS = IDs.id(IDs.FEATURE_SPACE, "controls")

    def __init__(
        self,
        bse: object,
        state_labels: dict[int, str] | None = None,
        options: FeatureSpaceOptions | None = None,
        id_suffix: str = "",
    ):
        super().__init__(bse, state_labels)  # type: ignore[arg-type]
        self.options = options or FeatureSpaceOptions()
        self.id_suffix = id_suffix

        # Override class-level IDs with instance-specific ones
        if id_suffix:
            self.X_SELECT = f"{IDs.id(IDs.FEATURE_SPACE, 'x-select')}-{id_suffix}"
            self.Y_SELECT = f"{IDs.id(IDs.FEATURE_SPACE, 'y-select')}-{id_suffix}"
            self.PLOT = f"{IDs.id(IDs.FEATURE_SPACE, 'plot')}-{id_suffix}"
            self.CONTROLS = f"{IDs.id(IDs.FEATURE_SPACE, 'controls')}-{id_suffix}"

    @property
    def page_id(self) -> str:
        return IDs.FEATURE_SPACE

    @property
    def nav_label(self) -> str:
        return "Feature Space"

    @property
    def nav_icon(self) -> str:
        return "📈"

    def get_feature_options(self, use_filtered: bool = True) -> list[dict[str, str]]:
        """Get dropdown options for feature selection.

        Returns a list of options with human-readable labels combining
        state labels and feature names (e.g., "ω - log_delta").

        Args:
            use_filtered: If True, use filtered features. If False, use extracted features.
        """
        if self.bse.solution is None:
            return [{"value": "0", "label": "Feature 0"}]

        # Determine which feature names to use
        if use_filtered and self.bse.solution.filtered_feature_names is not None:
            feature_names = self.bse.solution.filtered_feature_names
        elif not use_filtered and self.bse.solution.extracted_feature_names is not None:
            feature_names = self.bse.solution.extracted_feature_names
        else:
            return [{"value": "0", "label": "Feature 0"}]

        # Format feature names with state labels
        options: list[dict[str, str]] = []
        for idx, raw_name in enumerate(feature_names):
            parts = raw_name.split("__", 1)
            if len(parts) == 2 and parts[0].startswith("state_"):
                try:
                    state_idx = int(parts[0].replace("state_", ""))
                    state_label = self.get_state_label(state_idx)
                    feature_name = parts[1]
                    display_label = f"{state_label} - {feature_name}"
                except ValueError:
                    display_label = raw_name
            else:
                display_label = raw_name

            options.append({"value": str(idx), "label": display_label})

        return options if options else [{"value": "0", "label": "Feature 0"}]

    def get_n_features(self, use_filtered: bool = True) -> int:
        """Get the number of available features.

        Args:
            use_filtered: If True, count filtered features. If False, count extracted features.
        """
        if self.bse.solution is None:
            return 0

        if use_filtered and self.bse.solution.features is not None:
            return self.bse.solution.features.shape[1]
        elif not use_filtered and self.bse.solution.extracted_features is not None:
            return self.bse.solution.extracted_features.shape[1]

        return 0

    def get_label_options(self) -> list[dict[str, str]]:
        """Get dropdown options for label/cluster selection."""
        if self.bse.solution is None or self.bse.solution.labels is None:
            return []
        unique_labels = np.unique(self.bse.solution.labels)
        return [{"value": str(label), "label": str(label)} for label in unique_labels]

    def get_all_labels(self) -> list[str]:
        """Get all unique labels as strings."""
        if self.bse.solution is None or self.bse.solution.labels is None:
            return []
        unique_labels = np.unique(self.bse.solution.labels)
        return [str(label) for label in unique_labels]

    def build_layout(self) -> html.Div:
        """Build complete page layout with controls and plot.

        :return: Div containing controls and scatter plot.
        """
        use_filtered = self.options.use_filtered
        feature_options = self.get_feature_options(use_filtered=use_filtered)
        n_features = self.get_n_features(use_filtered=use_filtered)
        all_labels = self.get_all_labels()

        # Apply label filtering from options
        selected_labels = self.options.filter_labels(all_labels)

        x_feature = self.options.x_feature
        y_feature = self.options.y_feature

        y_select_style = {"display": "block"} if n_features > 1 else {"display": "none"}
        y_value = str(y_feature) if y_feature is not None and n_features > 1 else "0"

        feature_select_data = cast(Sequence[str], feature_options)

        # Show feature type switch only if extracted features exist
        show_feature_switch = (
            self.bse.solution is not None
            and self.bse.solution.extracted_features is not None
            and self.bse.solution.features is not None
        )

        return html.Div(
            [
                # Controls panel
                dmc.Paper(
                    [
                        dmc.Group(
                            [
                                dmc.Select(
                                    id=self.X_SELECT,
                                    label="X Axis",
                                    data=feature_select_data,
                                    value=str(x_feature),
                                    w=200,
                                ),
                                html.Div(
                                    dmc.Select(
                                        id=self.Y_SELECT,
                                        label="Y Axis",
                                        data=feature_select_data,
                                        value=y_value,
                                        w=200,
                                    ),
                                    id=self.Y_CONTAINER,
                                    style=y_select_style,
                                ),
                                dmc.MultiSelect(
                                    id=self.LABEL_SELECT,
                                    label="Labels",
                                    data=all_labels,
                                    value=selected_labels,
                                    w=300,
                                ),
                                dmc.Switch(
                                    id=self.FEATURE_TYPE_SWITCH,
                                    label="Use Filtered Features",
                                    checked=use_filtered,
                                    size="md",
                                )
                                if show_feature_switch
                                else html.Div(),
                            ],
                            gap="md",
                        ),
                    ],
                    p="md",
                    mb="md",
                    withBorder=True,
                    id=self.CONTROLS,
                ),
                # Main plot
                dmc.Paper(
                    [
                        dcc.Graph(
                            id=self.PLOT,
                            figure=self.build_figure(
                                x_feature=x_feature,
                                y_feature=y_feature,
                                selected_labels=selected_labels,
                            ),
                            style={
                                "height": "70vh",
                                "aspectRatio": "1 / 1",
                                "maxWidth": "70vh",
                                "margin": "0 auto",
                            },
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
        feature_options = self.get_feature_options()
        n_features = self.get_n_features()

        controls: list[dmc.Select] = []
        feature_select_data = cast(Sequence[str], feature_options)

        controls.append(
            dmc.Select(
                id=self.X_SELECT,
                label="X Axis",
                data=feature_select_data,
                value="0",
                w=200,
            )
        )

        if n_features > 1:
            controls.append(
                dmc.Select(
                    id=self.Y_SELECT,
                    label="Y Axis",
                    data=feature_select_data,
                    value="1" if n_features > 1 else "0",
                    w=200,
                )
            )

        return dmc.Group(controls, gap="md")

    def build_figure(
        self,
        x_feature: int = 0,
        y_feature: int | None = None,
        selected_labels: list[str] | None = None,
        use_filtered: bool = True,
        **kwargs: object,
    ) -> go.Figure:
        """Build the feature space scatter plot.

        Args:
            x_feature: Feature index for x-axis.
            y_feature: Feature index for y-axis. If None, creates a 1D strip plot.
            selected_labels: List of label strings to display. If None, shows all.
            use_filtered: If True, use filtered features. If False, use extracted features.
        """
        if self.bse.solution is None:
            return go.Figure()

        # Select which features to use
        if use_filtered and self.bse.solution.features is not None:
            features = self.bse.solution.features.cpu().numpy()
        elif not use_filtered and self.bse.solution.extracted_features is not None:
            features = self.bse.solution.extracted_features.cpu().numpy()
        else:
            return go.Figure()

        labels = np.array(self.bse.solution.labels)
        unique_labels = np.unique(labels)
        n_features = features.shape[1]

        # Filter to selected labels
        if selected_labels is not None:
            unique_labels = np.array(
                [label for label in unique_labels if str(label) in selected_labels]
            )

        feature_options = self.get_feature_options(use_filtered=use_filtered)

        x_label = (
            feature_options[x_feature]["label"]
            if x_feature < len(feature_options)
            else f"Feature {x_feature}"
        )

        if n_features == 1 or y_feature is None:
            return self._build_strip_plot(features, labels, unique_labels, x_feature, x_label)

        y_label = (
            feature_options[y_feature]["label"]
            if y_feature < len(feature_options)
            else f"Feature {y_feature}"
        )
        return self._build_2d_scatter(
            features, labels, unique_labels, x_feature, y_feature, x_label, y_label
        )

    def _build_strip_plot(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        unique_labels: np.ndarray,
        x_feature: int,
        x_label: str,
    ) -> go.Figure:
        """Build a 1D strip plot when only one feature is available."""
        fig = go.Figure()
        scatter_type = go.Scattergl if use_webgl(self.bse.y0) else go.Scatter

        for i, label in enumerate(unique_labels):
            idx = np.where(labels == label)[0]
            jitter = np.random.uniform(-0.3, 0.3, size=len(idx))

            fig.add_trace(
                scatter_type(
                    x=features[idx, x_feature],
                    y=jitter,
                    mode="markers",
                    marker={"size": 6, "color": get_color(i), "opacity": 0.6},
                    name=str(label),
                    customdata=idx,
                    hovertemplate=(
                        f"{x_label}: %{{x:.4f}}<br>Index: %{{customdata}}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title="Feature Space (1D Strip Plot) - click to inspect",
            xaxis_title=x_label,
            yaxis={"visible": False, "range": [-1, 1]},
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
            hovermode="closest",
        )

        return fig

    def _build_2d_scatter(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        unique_labels: np.ndarray,
        x_feature: int,
        y_feature: int,
        x_label: str,
        y_label: str,
    ) -> go.Figure:
        """Build a 2D scatter plot for feature space."""
        fig = go.Figure()
        scatter_type = go.Scattergl if use_webgl(self.bse.y0) else go.Scatter

        for i, label in enumerate(unique_labels):
            idx = np.where(labels == label)[0]
            fig.add_trace(
                scatter_type(
                    x=features[idx, x_feature],
                    y=features[idx, y_feature],
                    mode="markers",
                    marker={"size": 4, "color": get_color(i), "opacity": 0.6},
                    name=str(label),
                    customdata=idx,
                    hovertemplate=(
                        f"{x_label}: %{{x:.4f}}<br>"
                        f"{y_label}: %{{y:.4f}}<br>"
                        "Index: %{customdata}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title="Feature Space (click to inspect)",
            xaxis_title=x_label,
            yaxis_title=y_label,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
            hovermode="closest",
        )

        return fig


# Global instance placeholder - will be set by InteractivePlotter
_page_instance: FeatureSpacePage | None = None


def set_page_instance(page: FeatureSpacePage) -> None:
    """Set the global page instance for callbacks."""
    global _page_instance
    _page_instance = page


@callback(
    Output(FeatureSpacePage.PLOT, "figure"),
    [
        Input(FeatureSpacePage.X_SELECT, "value"),
        Input(FeatureSpacePage.Y_SELECT, "value"),
        Input(FeatureSpacePage.LABEL_SELECT, "value"),
        Input(FeatureSpacePage.FEATURE_TYPE_SWITCH, "checked"),
    ],
    prevent_initial_call=True,
)
def update_feature_space_figure(
    x_feature: str, y_feature: str, selected_labels: list[str], use_filtered: bool | None
) -> go.Figure:
    """Update figure when axis or label selection changes."""
    if _page_instance is None:
        return go.Figure()

    # Default to True if switch doesn't exist
    if use_filtered is None:
        use_filtered = True

    n_features = _page_instance.get_n_features(use_filtered=use_filtered)
    y_feat = int(y_feature) if n_features > 1 else None
    return _page_instance.build_figure(
        x_feature=int(x_feature),
        y_feature=y_feat,
        selected_labels=selected_labels,
        use_filtered=use_filtered,
    )


@callback(
    [
        Output(TrajectoryModal.MODAL, "opened", allow_duplicate=True),
        Output(TrajectoryModal.MODAL, "title", allow_duplicate=True),
        Output(TrajectoryModal.MODAL_INFO, "children", allow_duplicate=True),
        Output(SELECTED_SAMPLE_DATA, "data", allow_duplicate=True),
    ],
    Input(FeatureSpacePage.PLOT, "clickData"),
    [
        State(FeatureSpacePage.X_SELECT, "value"),
        State(FeatureSpacePage.Y_SELECT, "value"),
    ],
    prevent_initial_call=True,
)
def open_trajectory_modal_from_feature_space(
    click_data: dict[str, Any] | None,
    x_feature: str,
    y_feature: str,
) -> tuple[
    bool | NoUpdate,
    str | NoUpdate,
    str | NoUpdate,
    dict[str, Any] | None | NoUpdate,
]:
    """Open trajectory modal when a point is clicked."""
    if _page_instance is None or click_data is None:
        return no_update, no_update, no_update, no_update

    bse = _page_instance.bse
    if bse.solution is None or bse.solution.labels is None:
        return no_update, no_update, no_update, no_update

    try:
        point = click_data["points"][0]
        curve_number = point["curveNumber"]
        point_index = point["pointIndex"]

        labels = np.array(bse.solution.labels)
        unique_labels = np.unique(labels)

        if curve_number >= len(unique_labels):
            return no_update, no_update, no_update, no_update

        target_label = unique_labels[curve_number]
        label_indices = np.where(labels == target_label)[0]

        if point_index >= len(label_indices):
            return no_update, no_update, no_update, no_update

        sample_idx = int(label_indices[point_index])
        label = bse.solution.labels[sample_idx]

        x_coord = point.get("x", 0)
        y_coord = point.get("y", 0)

        feature_options = _page_instance.get_feature_options()
        x_label = (
            feature_options[int(x_feature)]["label"]
            if int(x_feature) < len(feature_options)
            else f"Feature {x_feature}"
        )
        y_label = (
            feature_options[int(y_feature)]["label"]
            if int(y_feature) < len(feature_options)
            else f"Feature {y_feature}"
        )

        title = f"Trajectory: Sample {sample_idx} (Label: {label})"
        info = f"Features: {x_label} = {x_coord:.4f}, {y_label} = {y_coord:.4f}"

        return (
            True,
            title,
            info,
            {"sample_idx": sample_idx, "label": str(label)},
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return no_update, no_update, no_update, no_update
